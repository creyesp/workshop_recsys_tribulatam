from typing import Dict
from typing import Text

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


class TwoTowerModel(tfrs.Model):
    def __init__(
        self,
        query_model: tf.keras.Model,
        candidate_model: tf.keras.Model,
        task: tfrs.tasks.Retrieval,
        compute_metrics=False,
    ):
        super().__init__()
        self.query_model = query_model
        self.candidate_model = candidate_model
        self.task: tfrs.tasks.Retrieval = task
        self.compute_metrics = compute_metrics

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        if len(self.query_model.inputs) == 1:
            query_input = features[self.query_model.inputs[0].name]
        else:
            query_input = {x.name: features[x.name] for x in self.query_model.inputs}

        if len(self.candidate_model.inputs) == 1:
            candidate_input = features[self.candidate_model.inputs[0].name]
        else:
            candidate_input = {x.name: features[x.name] for x in self.candidate_model.inputs}

        query_embeddings = self.query_model(query_input)
        candidate_embeddings = self.candidate_model(candidate_input)

        return self.task(query_embeddings, candidate_embeddings, compute_metrics=not training)


def train():
    data_path = "../data/ta_feng_all_months_merged.zip"
    data = pd.read_csv(data_path, compression="zip")
    data["AGE_GROUP"] = data["AGE_GROUP"].backfill()

    customer_ids = data["CUSTOMER_ID"].unique().tolist()
    product_ids = data["PRODUCT_ID"].unique().tolist()

    fetures = ["CUSTOMER_ID", "PRODUCT_ID"]
    data["DT"] = pd.to_datetime(data["TRANSACTION_DT"])
    data["validation_group"] = data.groupby("CUSTOMER_ID")["DT"].rank(method="first", ascending=False) == 1
    min_order_by_user = data.groupby("CUSTOMER_ID")["CUSTOMER_ID"].transform("count") > 4
    interaction_train = data.loc[min_order_by_user & ~data["validation_group"], fetures]
    interaction_val = data.loc[min_order_by_user & data["validation_group"], fetures]

    candidates = interaction_train[["PRODUCT_ID"]].drop_duplicates()

    batch_size = 1024 * 8
    tf.random.set_seed(42)
    train_ds_raw = tf.data.Dataset.from_tensor_slices(interaction_train.to_dict(orient="list"))
    train_ds_raw = train_ds_raw.shuffle(buffer_size=100_000, reshuffle_each_iteration=True).batch(batch_size)
    val_ds_raw = tf.data.Dataset.from_tensor_slices(interaction_val.to_dict(orient="list"))
    val_ds_raw = val_ds_raw.batch(batch_size)

    candidate_ds_raw = tf.data.Dataset.from_tensor_slices(candidates.to_dict(orient="list"))

    ######################

    lookup_customer = tf.keras.layers.IntegerLookup(vocabulary=customer_ids)
    lookup_product = tf.keras.layers.IntegerLookup(vocabulary=product_ids)

    train_ds = train_ds_raw.map(
        lambda x: {
            "CUSTOMER_ID": lookup_customer(x["CUSTOMER_ID"]),
            "PRODUCT_ID": lookup_product(x["PRODUCT_ID"]),
        }
    ).cache()

    val_ds = val_ds_raw.map(
        lambda x: {
            "CUSTOMER_ID": lookup_customer(x["CUSTOMER_ID"]),
            "PRODUCT_ID": lookup_product(x["PRODUCT_ID"]),
        }
    ).cache()

    candidate_ds = candidate_ds_raw.map(lambda x: {"PRODUCT_ID": lookup_product(x["PRODUCT_ID"])}).cache()

    customer_emb_dim = 128 * 4
    product_emb_dim = 128 * 4

    epochs = 200
    learning_rate = 0.0001

    query_model_base = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(), name="CUSTOMER_ID", dtype=tf.dtypes.int64),
            tf.keras.layers.Embedding(
                input_dim=lookup_customer.vocabulary_size(), output_dim=customer_emb_dim, embeddings_regularizer="l2"
            ),
        ]
    )

    candidate_model_base = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(), name="PRODUCT_ID", dtype=tf.dtypes.int64),
            tf.keras.layers.Embedding(
                input_dim=lookup_product.vocabulary_size(), output_dim=product_emb_dim, embeddings_regularizer="l2"
            ),
        ]
    )

    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(candidates=candidate_ds.batch(batch_size).map(candidate_model_base))
    )

    model = TwoTowerModel(query_model_base, candidate_model_base, task, compute_metrics=False)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate * 10))
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_factorized_top_k/top_100_categorical_accuracy", patience=2, restore_best_weights=True
            ),
        ],
    )

    export_query_model = tf.keras.models.Sequential(
        [
            query_model_base.input,
            lookup_customer,
            query_model_base,
            tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1)),
        ]
    )

    export_candidate_model = tf.keras.models.Sequential(
        [
            candidate_model_base.input,
            lookup_product,
            candidate_model_base,
            tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1)),
        ]
    )

    bruteforce_index = tfrs.layers.factorized_top_k.BruteForce(
        query_model=export_query_model, k=10, name="index_bruteforce"
    )

    bruteforce_index.index_from_dataset(
        tf.data.Dataset.zip(
            (
                candidate_ds_raw.batch(100).map(lambda x: x["PRODUCT_ID"]),
                candidate_ds_raw.batch(100).map(lambda x: x["PRODUCT_ID"]).map(export_candidate_model),
            )
        )
    )

    example = {"CUSTOMER_ID": tf.constant([1477962])}

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                [
                    None,
                ],
                dtype=tf.int64,
                name="CUSTOMER_ID",
            ),
        ]
    )
    def signature_default(customer_id):
        score, products = bruteforce_index.call({"CUSTOMER_ID": customer_id})
        return {
            "scores": score,
            "items": products,
        }

    # to save as keras model is necessary to call the model before save it
    bruteforce_index(example)
    output_path = "../models/two_tower"
    tf.keras.models.save_model(
        bruteforce_index,
        output_path,
        signatures={
            "serving_default": signature_default,
        },
    )


if __name__ == "__main__":
    train()
