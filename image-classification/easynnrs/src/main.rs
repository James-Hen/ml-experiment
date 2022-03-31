extern crate easynn;
extern crate rand;
extern crate rust_mnist;

use easynn::prelude::*;
use rand::Rng;
use rust_mnist::{print_sample_image, Mnist};

fn main() {
    // create the network and add 4 layers
    let mut nn = Sequential::<f64>::new(Loss::MeanSquare);
    // add the input (2 integers) and a hidden layer
    nn.add(Dense::<f64>::new(sh!([28, 28]), sh!([256]), Activation::Relu));
    // add another hidden layer
    nn.add(Dense::<f64>::new(sh!([256]), sh!([128]), Activation::Relu));
    // add another hidden layer
    nn.add(Dense::<f64>::new(sh!([128]), sh!([64]), Activation::Relu));
    // add the output layer
    nn.add(Dense::<f64>::new(sh!([64]), sh!([10]), Activation::Relu));

    // create the training set
    let mnist = Mnist::new("../data/FashionMNIST/raw/");
    // Print one image (the one at index 5) for verification.
    // print_sample_image(&mnist.train_data[5], mnist.train_labels[5]);
    let mut inputs = Vec::<Tensor<f64>>::new();     // images
    let mut outputs = Vec::<Tensor<f64>>::new();    // one-hot classifications
    for im in mnist.train_data {
        let mut im_f = Vec::<f64>::new();
        for x in im {
            im_f.push(x.into());
        }
        inputs.push(Tensor::new(sh!([28, 28]), im_f));
    }
    for cl in mnist.train_labels {
        let mut cl_hot = Tensor::<f64>::zeros(sh!([10]));
        cl_hot.set([cl as usize], 1.);
        inputs.push(cl_hot);
    }

    // train the model
    for _i in 0..10 {
        nn.train_once(&inputs, &outputs, 100, 0.1, true);
    }

    // evaluate the model
    /*let test_in1: u32 = 19260817;
    let test_in2: u32 = 1145141919;
    let test_out: u32 = target_func(test_in1, test_in2);
    let test_res = nn.predict(&Tensor::new(sh!([2]), vec![test_in1 as f64, test_in2 as f64])).unwrap().get([0]);
    println!("The prediction of input\n\t{:b} and\n\t{:b} is\n\t{:b} , expected\n\t{:b}"
        , test_in1, test_in2, test_res.floor() as u32, test_out);*/
}