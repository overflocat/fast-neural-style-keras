For training the network:

```
python main.py -c ./configs/wave.json -m train -i
```

For predicting:

```
python main.py -c ./configs/wave.json -m predict -i ./examples/content_img/content_2.png -o ./res.jpg
```

For viewing the baseline:

```
python main.py -c ./configs/wave.json -m temp_view -i ./examples/content_img/content_2.png -o ./res.jpg --iters 600
```

A more detailed README will be added later.