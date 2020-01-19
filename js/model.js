const H5_MODEL_URL = "mnist_tfjs_layers_model/model.json";
const SAVED_MODEL_URL = "mnist_tfjs_graph_model/model.json";

let mnist_model = null;
const IMAGE_SIZE = 280;
const CROPPED_IMAGE_SIZE = 28;

function load_model(error, callback) {
    tf.loadLayersModel(H5_MODEL_URL).then(model => {
        mnist_model = model;
        callback(model);
    }).catch(e => {
        error(e);
    });
}

function classify() {
    const bitmap = getImageData();
    const sample = [];
    let canvas2 = document.getElementById("canvas2");
    let ctx = canvas2.getContext("2d");
    let myImageData = ctx.createImageData(CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE);
    for (let i = 0; i < CROPPED_IMAGE_SIZE; i++) {
        const row = [];
        for (let j = 0; j < CROPPED_IMAGE_SIZE; j++) {
            let b = get_value(bitmap, i, j);
            row.push(b);
            if (b != 0) {
                myImageData.data[(i * CROPPED_IMAGE_SIZE + j) * 4] = 255;
                myImageData.data[(i * CROPPED_IMAGE_SIZE + j) * 4 + 3] = 255;
            }
        }
        sample.push(row);
    }
    ctx.putImageData(myImageData, 0, 0);

    const input = tf.tensor([sample], shape=[1, 28, 28], dtype="float32");
    const output = mnist_model.predict(input).dataSync();
    let label, prob = 0;
    for (let i in output) {
        if (output[i] > prob) {
            prob = output[i];
            label = i;
        }
    }
    if (prob > 0.5) {
        document.getElementById("info").innerHTML = `You drew a ${label}. I am ${Math.round(prob * 100)}% sure.`;
    } else {
        document.getElementById("info").innerHTML = "I was unable to identify what digit you wrote.";
    }
}

function get_value(bitmap, i, j) {
    for (let k = 0; k < 10; k++) {
        for (let l = 0; l < 10; l++) {
            for (let m = 0; m < 4; m++) {
                if (bitmap[((i * 10 + k) * IMAGE_SIZE + j * 10 + 1) * 4 + m] != 0) {
                    return 1.0;
                }
            }
        }
    }
    return 0.0;
}