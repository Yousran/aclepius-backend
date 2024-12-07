const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const moment = require('moment');
const { Storage } = require('@google-cloud/storage');
const { Firestore } = require('@google-cloud/firestore');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const port = process.env.PORT || 8080; // Use the PORT environment variable or default to 8080

// Initialize Cloud Storage and Firestore
const storage = new Storage();
const firestore = new Firestore();
const bucketName = 'bucket-submissionmlgc-amyusran';
const modelPath = 'model.json';

// Load the TensorFlow.js model from Cloud Storage
let model;
async function loadModel() {
  model = await tf.loadGraphModel('https://storage.googleapis.com/bucket-submissionmlgc-amyusran/model.json');
}
loadModel();


// Set up multer for file upload handling
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 1000000 }, // 1MB limit
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith('image/')) {
      return cb(new Error('Only image files are allowed!'), false);
    }
    cb(null, true);
  }
}).single('image');

app.post('/predict', (req, res) => {
  upload(req, res, async function (err) {
    if (err instanceof multer.MulterError) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
          status: 'fail',
          message: 'Payload content length greater than maximum allowed: 1000000'
        });
      }
      return res.status(400).json({
        status: 'fail',
        message: 'Terjadi kesalahan dalam melakukan prediksi'
      });
    } else if (err) {
      return res.status(400).json({
        status: 'fail',
        message: 'Terjadi kesalahan dalam melakukan prediksi'
      });
    }

    if (!req.file) {
      return res.status(400).json({
        status: 'fail',
        message: 'No file uploaded'
      });
    }

    // Preprocess the image
    const imageBuffer = req.file.buffer;
    const imageTensor = tf.node.decodeImage(imageBuffer, 3)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat()
      .div(tf.scalar(255.0));

    // Make prediction
    const prediction = model.predict(imageTensor);
    const predictionValue = prediction.dataSync()[0];
    const isCancer = predictionValue > 0.5;
    const result = isCancer ? 'Cancer' : 'Non-cancer';
    const suggestion = isCancer ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';

    const response = {
      status: 'success',
      message: 'Model is predicted successfully',
      data: {
        id: uuidv4(),
        result: result,
        suggestion: suggestion,
        createdAt: moment().toISOString()
      }
    };

    // Save prediction to Firestore
    await firestore.collection('predictions').doc(response.data.id).set(response.data);

    res.status(201).json(response);
  });
});

app.get('/predict/histories', async (req, res) => {
  const snapshot = await firestore.collection('predictions').get();
  const histories = snapshot.docs.map(doc => doc.data());

  res.status(200).json({
    status: 'success',
    data: histories
  });
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Server is running on port ${port}`);
});