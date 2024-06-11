package com.example.finalprojectpapb;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private ImageView imageView;
    private TextView resultTextView;
    private Interpreter tflite;
    private Bitmap selectedImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.acneimage);
        resultTextView = findViewById(R.id.result);
        Button detectButton = findViewById(R.id.detectbutton);

        try {
            tflite = new Interpreter(loadModelFile());
            Log.d(TAG, "Model loaded successfully");
        } catch (IOException e) {
            Log.e(TAG, "Failed to load model", e);
        }

        selectedImage = BitmapFactory.decodeResource(getResources(), R.drawable.acne2);
        imageView.setImageBitmap(selectedImage);

        detectButton.setOnClickListener(v -> {
            if (selectedImage != null) {
                new AcneDetectionTask().execute(selectedImage);
            } else {
                Log.e(TAG, "No image selected");
            }
        });
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("mobilenet_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private class AcneDetectionTask extends AsyncTask<Bitmap, Void, String> {
        @Override
        protected String doInBackground(Bitmap... bitmaps) {
            if (tflite == null) {
                Log.e(TAG, "TensorFlow Lite interpreter is not initialized");
                return "Error: Model not loaded";
            }

            Bitmap bitmap = bitmaps[0];
            float[][][][] input = preprocessImage(bitmap);
            float[][] output = new float[1][4];
            tflite.run(input, output);

            boolean acneDetected = false;
            for (float value : output[0]) {
                if (value > 0.5) {
                    acneDetected = true;
                    break;
                }
            }
            return acneDetected ? "Acne Detected" : "Acne Not Detected";
        }

        @Override
        protected void onPostExecute(String result) {
            resultTextView.setText(result);
        }

        private float[][][][] preprocessImage(Bitmap bitmap) {
            // Resize and normalize the image
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            int width = resizedBitmap.getWidth();
            int height = resizedBitmap.getHeight();
            int[] intValues = new int[width * height];
            resizedBitmap.getPixels(intValues, 0, width, 0, 0, width, height);

            float[][][][] floatValues = new float[1][224][224][3];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixel = intValues[y * width + x];
                    floatValues[0][y][x][0] = ((pixel >> 16) & 0xFF) / 255.0f;
                    floatValues[0][y][x][1] = ((pixel >> 8) & 0xFF) / 255.0f;
                    floatValues[0][y][x][2] = (pixel & 0xFF) / 255.0f;
                }
            }
            return floatValues;
        }
    }
}
