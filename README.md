# Chinese_Poem_RNN
##General
Using RNN to generate 藏头诗, the idea is exactly the same with Andrej Karpathy's [Char-RNN](https://github.com/karpathy/char-rnn), but work on Chinese poems data. We use Quan Tangshi as the training data(nearly 60% used). 

##Sample Output
```
1)                      2)                    3)
卧风风雨落，            新府高南苑，          卧山春色远，
石叶晚风明。            年年未有人。          石里夜烟深。
沙上春风晚，            快衣王尺石，          沙上云前里，
壁中江水深。            乐马海中州。          壁随风上人。
```

##Model
Use the previous 32 chars to predict the next char, use 0-ahead padding to transform it to fixed input. Chars is encoded by one-hot encoding, there are around ~6000 chars in the training data. The model stacked 2 LSTM modules, each with 512 neurons. And 0.2 dropout rate while training. 

##Setup
* 1) `python prepare.py` to generate the training data
* 2) `python model.py` to train the RNN model
* 3) The last three lines in `model.py` are for generating texts.

##TODO
Add the rhythming(押韵) functionality
