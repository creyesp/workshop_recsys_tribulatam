```commandline
docker run -it --rm -p 8501:8501 \
     -v "$(pwd)/models/:/models/" tensorflow/serving \
     --model_config_file=/models/models.config
```


```commandline
curl -X POST http://localhost:8501/v1/models/two_tower:predict \
  --header "Content-Type: application/json" \
  --data '{"instances": [{"CUSTOMER_ID": 1658781}]}'
```

```commandline
curl -X POST http://localhost:8501/v1/models/two_tower_context:predict \
  --header "Content-Type: application/json" \
  --data '{"instances": [{"CUSTOMER_ID": 1658781, "AGE_GROUP": "30-34"}]}'
```

```commandline
curl http://localhost:8501/v1/models/two_tower_context
```




'<25'
'25-29'
'30-34'
'35-39'
'40-44'
'45-49'
'50-54'
'55-59'
'60-64'
'>65'