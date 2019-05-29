package com.beawan;

import org.apache.log4j.Logger;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class Test2 {

    private static final Logger logger=Logger.getLogger(Test2.class);

    public Test2(String dllPath){
        System.load(dllPath);
    }

    public static String imgPath="F:\\opencvPhoto\\photo\\33.jpg";
    public static String savePath="F:\\opencvPhoto\\result\\";

    public static void main(String[] args) {

        Test2 test2=new Test2("D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");

        //读入图片
        Mat src = Imgcodecs.imread(imgPath);

        Mat gray=gray(src.clone());

        Mat canny=canny(gray.clone());

        Mat adaptiveThreshold=adaptiveThreshold(gray.clone());

        Mat dilate=dilate(adaptiveThreshold.clone());

        List<MatOfPoint> contours=getContours(canny.clone());

        drawContours(src.clone(),contours);

    }

    /**
     * 灰度化
     * @param src 原图
     * @return
     */
    public static Mat gray(Mat src){
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        return src;
    }

    /**
     * 边缘检测
     * @param src 灰度图
     * @return
     */
    public static Mat canny(Mat src) {
        //平滑处理
        Mat blur = new Mat();
        Imgproc.blur(src, blur, new Size(2, 2));

        //轮廓
        //使用Canndy检测边缘
        double lowThresh = 350;//双阀值抑制中的低阀值
        double heightThresh = 400;//双阀值抑制中的高阀值
        Mat cannyMat = new Mat();
//        Imgproc.Canny(blur, cannyMat,lowThresh, heightThresh,3,true);
        Imgproc.Canny(blur, cannyMat, lowThresh, heightThresh);

        //膨胀
        Mat dilate = new Mat();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));
        Imgproc.dilate(cannyMat, dilate, element, new Point(-1, -1), 1);

        Imgproc.threshold(dilate, dilate, 175, 255, Imgproc.THRESH_BINARY);

        Imgcodecs.imwrite(savePath+"canny.jpg", dilate);

        return dilate;
    }



    /**
     * 自适应二值化
     * @param src 灰度图
     * @return
     */
    public static Mat adaptiveThreshold(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C , Imgproc.THRESH_BINARY_INV, 25, 10);
        Imgproc.medianBlur(src, src, 5);
        Imgcodecs.imwrite(savePath+"adaptiveThreshold.jpg",src);
        return src;
    }

    /**
     * 膨胀
     * @param src 二值图
     * @return
     */
    public static Mat dilate(Mat src){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));
        Imgproc.dilate(src, src, element, new Point(-1, -1), 1);
        Imgcodecs.imwrite(savePath+"dilate.jpg",src);
        return src;
    }

    /**
     * 轮廓检测
     * @param src 二值膨胀图
     * @return
     */
    public static List<MatOfPoint> getContours(Mat src){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        List<MatOfPoint> resultContours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();

        //轮廓查找
        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_NONE);

        if(contours.size()<=0){
            logger.info("二值化后图片为全黑,不存在白色轮廓");
        }
        else{
            /*
            //寻找财报边缘(思路：财报的轮廓面积为最大)
            double maxArea=0;//最大轮廓面积
            for(int i=0;i<contours.size();i++){
                double contourArea=Imgproc.contourArea(contours.get(i));
                if(contourArea>300){
                    resultContours.add(contours.get(i));
                }
            }
            */
            return contours;
        }

        return null;
    }

    /**
     * 画轮廓
     * @param src 原图
     * @param contours 轮廓
     */
    public static void drawContours(Mat src,List<MatOfPoint> contours){
        List<MatOfPoint> list=new ArrayList<MatOfPoint>();
        Rect rect=null;
        for(MatOfPoint contour:contours){
            rect=Imgproc.boundingRect(contour);
            Imgproc.rectangle(src, rect, new Scalar(255, 0, 0), 1);
            list.add(contour);
//            if(Imgproc.contourArea(contour)/rect.area()>0.8) {
//                Imgproc.rectangle(src, rect, new Scalar(255, 0, 0), 1);
//                list.add(contour);
//            }
        }
        Imgproc.drawContours(src,list,-1,new Scalar(0,0,255),1);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\drawContours.jpg",src);
    }

}
