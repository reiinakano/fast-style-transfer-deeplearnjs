# Fast Neural Style Transfer in browser with Deeplearn.JS

This repository contains an implementation of the Fast Neural Style Transfer algorithm running fully inside a browser using the Deeplearn.JS library.

Demo website: https://reiinakano.github.io/fast-style-transfer-deeplearnjs

![demo_screen](demo_screen.png)

## FAQ

### What is this about?

This is an implementation of the Fast Neural Style Transfer algorithm running purely on the browser using the Deeplearn.JS library. Basically, a neural network attempts to "draw" one picture, the Content, in the style of another, the Style.

### Is my data safe? Can you see my webcam pics?

Your data and pictures here never leave your computer! In fact, this is one of the main advantages of running neural networks in your browser. Instead of sending us your data, we send *you* both the model *and* the code to run the model. These are then run by your browser.

### Is this implementation optimized?

As of today, there are major bottlenecks in the implementation of the model. Deeplearn.JS is a new library and some functions still do not have a WebGL implementation. Don't worry, as soon as the required functions become available, I'll update this site accordingly. I've [asked for help](https://github.com/PAIR-code/deeplearnjs/issues/141) on the official deeplearnjs repository, and will implement any advice I get for making this as fast as possible.

### How big are the models I'm downloading?

For each available style, your browser will download a model around ~6.6MB in size. Be careful if you have limited bandwidth (mobile data users).

### The web page is ugly.

I know. Sorry, I'm not really a UI designer. I have about a 10 minute tolerance for tweaking HTML and CSS until I give up. The good news is, it's all open source on Github! If you want to help improve the page's design, please send a pull request! :)

## Credits

Credits belong to the following:

* The authors of the original [Neural Style Transfer paper](https://arxiv.org/abs/1508.06576).
* The authors of the paper introducing [Real-Time Style Transfer](https://arxiv.org/abs/1603.08155).
* The author of the [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer) Github repository.
* The authors of [Deeplearn.JS](https://github.com/PAIR-code/deeplearnjs)
