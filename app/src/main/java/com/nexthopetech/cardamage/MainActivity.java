package com.nexthopetech.cardamage;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.nexthopetech.cardamage.ml.CarDamageModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, accuracy;
    ImageView imageView, btn_picture, about;

    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        accuracy = findViewById(R.id.accuracy);
        btn_picture = findViewById(R.id.picture);

        imageView = findViewById(R.id.imageView);

        about = findViewById(R.id.about);

        btn_picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //launch camera if permitted
                if(checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                }else{
                    //if not granted
                    requestPermissions(new String[] {Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode == 1 && resultCode == RESULT_OK){
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());

            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

            classifyImage(image);

        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void classifyImage(Bitmap image) {
        try {
            CarDamageModel model = CarDamageModel.newInstance(getApplicationContext());

            //create input for references
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect( 4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            //get 1d array of 224 * 224 pixels in image
            int[] intValue = new int[imageSize * imageSize];
            image.getPixels(intValue, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            //iterate over pixels and extract RGB values and add to bytebuffer
            int pixel = 0;
            for (int i = 0; i < imageSize; i++){
                for (int j = 0; j < imageSize; j++){
                    int val = intValue[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val  & 0xFF) * (1.f / 255.f));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            //run model interface and gets result

            CarDamageModel.Outputs outputs= model.process(inputFeature0);
            TensorBuffer outputFeatures0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidence = outputFeatures0.getFloatArray();

            //find the index of the class with the biggest confidence

            int maxPose = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidence.length; i++) {
                if(confidence[i] > maxConfidence){
                    maxConfidence = confidence[i];
                    maxPose = i;
                }
            }

            String[] classes = {"Cracks and Holes", "Medium Dents", "Severe Dents", "Severe Scratches", "Slightly Scratches", "Small Dents", "Windshield Damage"};
            result.setText(classes[maxPose]);


            //max confidence level
            float newMaxConfidence = maxConfidence * 100;
            int intMaxConfidence = (int) newMaxConfidence;

            accuracy.setText(classes[maxPose]+" "+String.valueOf(intMaxConfidence)+"%");
            //edited 27 -5- 2023
            /*for (int i = 0; i < confidence.length; i++) {
                accuracy.append("\n" + classes[i] + " " + String.valueOf(new_confidence[i]) + "\n");
            }*/

            //copy code of real show
            /*for (int i = 0; i < confidence.length; i++) {
                accuracy.append("\n"+classes[i]+" "+String.valueOf(confidence[i])+"\n");

            }*/



            result.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    //to search on internet
                    startActivity(new Intent(Intent.ACTION_VIEW,
                            Uri.parse("https://www.google.com/search?q="+result.getText())));
                }
            });

            model.close();

        }catch (IOException e){
            //to handle the exception

        }
    }


}