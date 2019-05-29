package com.beawan;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Canny {

    public Canny(String dllPath){
        System.load(dllPath);
    }

    public static void main(String[] args) {
        Canny canny=new Canny("D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");
        //读入图片
        Mat src = Imgcodecs.imread("F:\\opencvPhoto\\photo\\14.jpg");
        //将rgb灰化处理
        Mat gray=new Mat();
        Imgproc.cvtColor(src, gray,Imgproc.COLOR_BGR2GRAY);

        //平滑处理
        Mat blur=new Mat();
        Imgproc.blur(gray, blur, new Size(2, 2));

        //轮廓
        //使用Canndy检测边缘
        double lowThresh =100;//双阀值抑制中的低阀值
        double heightThresh = 300;//双阀值抑制中的高阀值
        Mat cannyMat=new Mat();
//        Imgproc.Canny(blur, cannyMat,lowThresh, heightThresh,3,true);
        Imgproc.Canny(blur, cannyMat,lowThresh, heightThresh);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\canny.jpg",cannyMat);

        //膨胀
        Mat dilate=new Mat();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(cannyMat, dilate, element, new Point(-1, -1), 1);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\dilate.jpg",dilate);

        Imgproc.threshold(dilate,dilate,175,255,Imgproc.THRESH_BINARY);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\cannyDilate.jpg",dilate);
        //黑白互换
        Core.bitwise_not(dilate,dilate);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\xx.jpg",dilate);

    }

}
