from typing import Dict
from typing import Text

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
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
    subclass_ids = data["PRODUCT_SUBCLASS"].unique().tolist()
    asset_ids = data["ASSET"].unique().tolist()
    age_group_ids = data["AGE_GROUP"].fillna("").unique().tolist()
    sale_price_min = data["SALES_PRICE"].min()
    sale_price_max = data["SALES_PRICE"].max()

    fetures = [
        "TRANSACTION_DT",
        "CUSTOMER_ID",
        "PRODUCT_ID",
        "PRODUCT_SUBCLASS",
        "ASSET",
        "AGE_GROUP",
        "SALES_PRICE",
    ]

    data["DT"] = pd.to_datetime(data["TRANSACTION_DT"])
    data["validation_group"] = data.groupby("CUSTOMER_ID")["DT"].rank(method="first", ascending=False) == 1
    min_order_by_user = data.groupby("CUSTOMER_ID")["CUSTOMER_ID"].transform("count") > 4
    intereaction_train = data.loc[min_order_by_user & ~data["validation_group"], fetures]
    intereaction_val = data.loc[min_order_by_user & data["validation_group"], fetures]

    candidates = (
        intereaction_train.groupby("PRODUCT_ID")
        .agg({"PRODUCT_SUBCLASS": "max", "ASSET": "max", "SALES_PRICE": "mean"})
        .reset_index()
    )

    batch_size = 1024 * 8
    tf.random.set_seed(42)
    train_ds_raw = tf.data.Dataset.from_tensor_slices(intereaction_train.to_dict(orient="list"))
    train_ds_raw = train_ds_raw.shuffle(buffer_size=100_000, reshuffle_each_iteration=True).batch(batch_size)
    val_ds_raw = tf.data.Dataset.from_tensor_slices(intereaction_val.to_dict(orient="list"))
    val_ds_raw = val_ds_raw.batch(batch_size)

    candidate_ds_raw = tf.data.Dataset.from_tensor_slices(candidates.to_dict(orient="list"))

    bucket_dt = 1000
    start_dt = tfa.text.parse_time("11/1/2000", "%m/%d/%Y", "SECOND")
    end_dt = tfa.text.parse_time("2/28/2001", "%m/%d/%Y", "SECOND")
    parser_dt = tf.keras.layers.Lambda(lambda x: tfa.text.parse_time(x, "%m/%d/%Y", "SECOND"))
    norm_datetime = tf.keras.layers.Lambda(
        lambda x: (tfa.text.parse_time(x, "%m/%d/%Y", "SECOND") - start_dt) / (end_dt - start_dt)
    )
    bucketize_datetime = tf.keras.layers.Discretization(bin_boundaries=tf.linspace(start_dt, end_dt, bucket_dt))
    lookup_customer = tf.keras.layers.IntegerLookup(vocabulary=customer_ids)
    lookup_product = tf.keras.layers.IntegerLookup(vocabulary=product_ids)
    lookup_subclass = tf.keras.layers.IntegerLookup(vocabulary=subclass_ids)
    lookup_asset = tf.keras.layers.IntegerLookup(vocabulary=asset_ids)
    lookup_age_group = tf.keras.layers.StringLookup(vocabulary=age_group_ids)
    norm_price = tf.keras.layers.Lambda(lambda x: (x - sale_price_min) / (sale_price_max - sale_price_min))

    train_ds = train_ds_raw.map(
        lambda x: {
            "TRANSACTION_DT": norm_datetime(x["TRANSACTION_DT"]),
            "BUCKETIZE_DT": bucketize_datetime(parser_dt(x["TRANSACTION_DT"])),
            "CUSTOMER_ID": lookup_customer(x["CUSTOMER_ID"]),
            "PRODUCT_ID": lookup_product(x["PRODUCT_ID"]),
            "PRODUCT_SUBCLASS": lookup_subclass(x["PRODUCT_SUBCLASS"]),
            "ASSET": lookup_asset(x["ASSET"]),
            "AGE_GROUP": lookup_age_group(x["AGE_GROUP"]),
            "SALES_PRICE": norm_price(x["SALES_PRICE"]),
        }
    )
    train_ds = train_ds.cache()

    val_ds = val_ds_raw.map(
        lambda x: {
            "TRANSACTION_DT": norm_datetime(x["TRANSACTION_DT"]),
            "BUCKETIZE_DT": bucketize_datetime(parser_dt(x["TRANSACTION_DT"])),
            "CUSTOMER_ID": lookup_customer(x["CUSTOMER_ID"]),
            "PRODUCT_ID": lookup_product(x["PRODUCT_ID"]),
            "PRODUCT_SUBCLASS": lookup_subclass(x["PRODUCT_SUBCLASS"]),
            "ASSET": lookup_asset(x["ASSET"]),
            "AGE_GROUP": lookup_age_group(x["AGE_GROUP"]),
            "SALES_PRICE": norm_price(x["SALES_PRICE"]),
        }
    )
    val_ds = val_ds.cache()

    candidate_ds = candidate_ds_raw.map(
        lambda x: {
            "PRODUCT_ID": lookup_product(x["PRODUCT_ID"]),
            "PRODUCT_SUBCLASS": lookup_subclass(x["PRODUCT_SUBCLASS"]),
            "ASSET": lookup_asset(x["ASSET"]),
            "SALES_PRICE": norm_price(x["SALES_PRICE"]),
        }
    ).cache()

    customer_emb_dim = 128 * 4
    product_emb_dim = 128 * 4
    age_group_emb_dim = 4
    subclass_emb_dim = 32

    asset_emb_dim = 32
    final_output_emb = 128 * 4

    epochs = 200
    learning_rate = 0.0001

    final_output_emb_factor = final_output_emb // 4

    query_customer_input = tf.keras.layers.Input(shape=(), name="CUSTOMER_ID", dtype=tf.dtypes.int64)
    query_age_input = tf.keras.layers.Input(
        shape=(),
        name="AGE_GROUP",
        dtype=tf.dtypes.int64,
    )
    # query_dt_input = tf.keras.layers.Input(shape=(1,), name="TRANSACTION_DT", dtype=tf.dtypes.float64, )
    query_customer_layer = tf.keras.layers.Embedding(
        input_dim=lookup_customer.vocabulary_size(), output_dim=customer_emb_dim, embeddings_regularizer="l2"
    )
    query_age_layer = tf.keras.layers.Embedding(
        input_dim=lookup_age_group.vocabulary_size(), output_dim=age_group_emb_dim, embeddings_regularizer="l2"
    )
    query_concat = [
        query_customer_layer(query_customer_input),
        query_age_layer(query_age_input),
        #   query_dt_input
    ]
    query_output = tf.keras.layers.Concatenate()(query_concat)
    query_output = tf.keras.layers.Dense(final_output_emb_factor, name="query_output_embedding")(query_output)
    query_model = tf.keras.models.Model(
        inputs=[
            query_customer_input,
            query_age_input,
            #      query_dt_input,
        ],
        outputs=query_output,
    )

    candidate_product_input = tf.keras.layers.Input(
        shape=(),
        name="PRODUCT_ID",
        dtype=tf.dtypes.int64,
    )
    candidate_subclass_input = tf.keras.layers.Input(
        shape=(),
        name="PRODUCT_SUBCLASS",
        dtype=tf.dtypes.int64,
    )
    candidate_asset_input = tf.keras.layers.Input(
        shape=(),
        name="ASSET",
        dtype=tf.dtypes.int64,
    )
    candidate_price_input = tf.keras.layers.Input(
        shape=(1,),
        name="SALES_PRICE",
        dtype=tf.dtypes.float64,
    )
    candidate_product_layer = tf.keras.layers.Embedding(
        input_dim=lookup_product.vocabulary_size(), output_dim=product_emb_dim, embeddings_regularizer="l2"
    )
    candidate_subclass_layer = tf.keras.layers.Embedding(
        input_dim=lookup_subclass.vocabulary_size(), output_dim=subclass_emb_dim, embeddings_regularizer="l2"
    )
    candidate_asset_layer = tf.keras.layers.Embedding(
        input_dim=lookup_asset.vocabulary_size(), output_dim=asset_emb_dim, embeddings_regularizer="l2"
    )
    candidate_concat = [
        candidate_product_layer(candidate_product_input),
        candidate_subclass_layer(candidate_subclass_input),
        candidate_asset_layer(candidate_asset_input),
        candidate_price_input,
    ]
    candidate_output = tf.keras.layers.Concatenate()(candidate_concat)
    candidate_output = tf.keras.layers.Dense(final_output_emb_factor, name="candidate_output_embedding")(
        candidate_output
    )
    candidate_model = tf.keras.models.Model(
        inputs=[candidate_product_input, candidate_subclass_input, candidate_asset_input, candidate_price_input],
        outputs=candidate_output,
    )

    # Define your objectives.
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(candidates=candidate_ds.batch(1024 * 8).map(candidate_model))
    )

    model = TwoTowerModel(query_model, candidate_model, task, compute_metrics=False)
    model.compile(
        # optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate/10)
        optimizer=tf.keras.optimizers.Adagrad()
    )
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

    query_customer_input = tf.keras.layers.Input(shape=(), name="CUSTOMER_ID", dtype=tf.dtypes.int64)
    query_age_input = tf.keras.layers.Input(
        shape=(),
        name="AGE_GROUP",
        dtype=tf.dtypes.string,
    )
    # query_dt_input = tf.keras.layers.Input(shape=(1,), name="TRANSACTION_DT", dtype=tf.dtypes.string, )

    query_customer_transform = lookup_customer(query_customer_input)
    query_age_transform = lookup_age_group(query_age_input)
    # query_dt_transform = norm_datetime(query_dt_input)
    query_output = query_model(
        [
            query_customer_transform,
            query_age_transform,
            # query_dt_transform
        ]
    )
    query_output = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(query_output)
    export_query_model = tf.keras.models.Model(
        inputs=[
            query_customer_input,
            query_age_input,
            # query_dt_input
        ],
        outputs=query_output,
    )

    candidate_product_input = tf.keras.layers.Input(
        shape=(),
        name="PRODUCT_ID",
        dtype=tf.dtypes.int64,
    )
    candidate_subclass_input = tf.keras.layers.Input(
        shape=(),
        name="PRODUCT_SUBCLASS",
        dtype=tf.dtypes.int64,
    )
    candidate_asset_input = tf.keras.layers.Input(
        shape=(),
        name="ASSET",
        dtype=tf.dtypes.int64,
    )
    candidate_price_input = tf.keras.layers.Input(
        shape=(1,),
        name="SALES_PRICE",
        dtype=tf.dtypes.float64,
    )

    candidate_product_transform = lookup_product(candidate_product_input)
    candidate_subclass_transform = lookup_subclass(candidate_subclass_input)
    candidate_asset_transform = lookup_asset(candidate_asset_input)
    candidate_price_transform = norm_price(candidate_price_input)

    candidate_output = candidate_model(
        [
            candidate_product_transform,
            candidate_subclass_transform,
            candidate_asset_transform,
            candidate_price_transform,
        ]
    )
    candidate_output = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(candidate_output)
    export_candidate_model = tf.keras.models.Model(
        inputs=[candidate_product_input, candidate_subclass_input, candidate_asset_input, candidate_price_input],
        outputs=candidate_output,
    )

    bruteforce_index = tfrs.layers.factorized_top_k.BruteForce(
        query_model=export_query_model, k=10, name="index_bruteforce"
    )

    bruteforce_index.index_from_dataset(
        tf.data.Dataset.zip(
            (
                candidate_ds_raw.batch(100).map(lambda x: x["PRODUCT_ID"]),
                candidate_ds_raw.batch(100).map(export_candidate_model),
            )
        )
    )

    example = {
        "CUSTOMER_ID": tf.constant([1658781]),
        "AGE_GROUP": tf.constant([b"30-34"]),
        #        "TRANSACTION_DT": tf.constant([b'11/5/2000']),
    }

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                [
                    None,
                ],
                dtype=tf.dtypes.int64,
                name="CUSTOMER_ID",
            ),
            tf.TensorSpec(
                [
                    None,
                ],
                dtype=tf.dtypes.string,
                name="AGE_GROUP",
            ),
            # tf.TensorSpec([None, ], dtype=tf.dtypes.string, name="TRANSACTION_DT"),
        ]
    )
    def signature_default(
        customer_id,
        age_group,
        # transaction_dt,
    ):
        score, products = bruteforce_index.call(
            {
                "CUSTOMER_ID": customer_id,
                "AGE_GROUP": age_group,
                #   "TRANSACTION_DT": transaction_dt,
            }
        )
        return {
            "scores": score,
            "items": products,
        }

    # to save as keras model is necessary to call the model before save it
    bruteforce_index(example)

    output_path = "../models/two_tower_context/1/"
    bruteforce_index.save(
        output_path,
        signatures={
            "serving_default": signature_default,
        },
    )


if __name__ == "__main__":
    train()
