package com.beawan;

import org.apache.log4j.Logger;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgproc.CV_FILLED;

public class Test {

    private static final Logger logger=Logger.getLogger(Test.class);

    public Test(String dllPath){
        System.load(dllPath);
    }

    public static String imgPath="F:\\opencvPhoto\\photo\\32.jpg";

    public static void main(String[] args) {

        Test test=new Test("D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");


        //读入图片
        Mat src = Imgcodecs.imread(imgPath);

        Imgproc.GaussianBlur(src, src, new Size(3,3), 0, 0, Core.BORDER_DEFAULT);


        Mat gray=gray(src.clone());
        Mat canny=canny(gray.clone());
        Mat adaptiveThreshold2=adaptiveThreshold2(gray.clone());
        Mat erode=erode(adaptiveThreshold2.clone());
        MatOfPoint contour=getContours(erode.clone());
        drawContours(src.clone(),contour);



        /*
        //灰度化
        Mat gray=gray(src.clone());

        Mat canny=canny(gray.clone());

        Mat adaptiveThreshold=adaptiveThreshold(gray.clone());
        Mat adaptiveThreshold2=adaptiveThreshold2(gray.clone());

        Mat adaptiveThreshold3=operate2(adaptiveThreshold.clone(),adaptiveThreshold2.clone());

        List<MatOfPoint> contours=getContours2(canny.clone());
        removeBackGround2(src.clone(),contours);

        drawContours2(src.clone(),contours);


        Mat operate=operate(adaptiveThreshold3.clone(),canny.clone());

        MatOfPoint contour=getContours(adaptiveThreshold.clone());
        removeBackGround(src.clone(),contour);

        drawContours(src.clone(),contour);
        Rect rect=getRect(contour);
        drawRect(src.clone(),rect);
        cutImg(src.clone(),rect);
        */


    }

    /**
     * 灰度化
     * @param src
     * @return
     */
    public static Mat gray(Mat src){
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        return src;
    }

    public static Mat canny(Mat gray){
        //平滑处理
        Mat blur=new Mat();
        Imgproc.blur(gray, blur, new Size(2, 2));

        //轮廓
        //使用Canndy检测边缘
        double lowThresh =350;//双阀值抑制中的低阀值
        double heightThresh = 350;//双阀值抑制中的高阀值
        Mat cannyMat=new Mat();
        Imgproc.Canny(blur, cannyMat,lowThresh, heightThresh,3,true);
//        Imgproc.Canny(blur, cannyMat,lowThresh, heightThresh);

        //膨胀
        Mat dilate=new Mat();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(cannyMat, dilate, element, new Point(-1, -1), 1);

        Imgproc.threshold(dilate,dilate,175,255,Imgproc.THRESH_BINARY);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\testCanny.jpg",dilate);

        return dilate;
    }

    /**
     * 黑白互换
     * @param canny
     * @return
     */
    public static Mat colorInterchange(Mat canny){
        Core.bitwise_not(canny,canny);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\colorInterchange.jpg",canny);
        return canny;
    }

    public static Mat adaptiveThreshold(Mat gray){
        Imgproc.adaptiveThreshold(gray, gray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C , Imgproc.THRESH_BINARY, 25, -5);
//        Imgproc.adaptiveThreshold(gray, gray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C , Imgproc.THRESH_BINARY, 25, 5);
        Imgproc.medianBlur(gray, gray, 3);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\testThreshold.jpg",gray);

        return gray;
    }

    public static Mat adaptiveThreshold2(Mat gray){
        Imgproc.adaptiveThreshold(gray, gray, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C , Imgproc.THRESH_BINARY, 25, 5);
//        Imgproc.adaptiveThreshold(gray, gray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C , Imgproc.THRESH_BINARY, 25, 5);
        Imgproc.medianBlur(gray, gray, 5);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\testThreshold2.jpg",gray);
        return gray;
    }

    /**
     * 腐蚀
     * @param src
     * @return
     */
    public static Mat erode(Mat src){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5));
        Imgproc.erode(src, src, element, new Point(-1, -1), 2);
        Imgproc.medianBlur(src, src, 3);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\erode.jpg",src);
        return  src;
    }

    /**
     *
     * @param src 二值化图
     * @return
     */
    public static MatOfPoint getContours(Mat src){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        MatOfPoint contour=null;
        Mat hierarchy = new Mat();

        Mat hole=src.clone();
        //轮廓查找
        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Mat temp = Imgcodecs.imread(imgPath);
        drawContours2(temp,contours);

        if(contours.size()<=0){
            logger.info("二值化后图片为全黑,不存在白色轮廓");
        }
        else{
            //寻找财报边缘(思路：财报的轮廓面积为最大)
            double maxArea=0;//最大轮廓面积
            for(int i=0;i<contours.size();i++){

                double contourArea=Imgproc.contourArea(contours.get(i));
                if(contourArea>maxArea){
                    maxArea=contourArea;
                    contour=contours.get(i);
                }
            }
        }

        return contour;
    }

    public static List<MatOfPoint> getContours2(Mat src){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        List<MatOfPoint> resultContours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();

        //轮廓查找
        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        if(contours.size()<=0){
            logger.info("二值化后图片为全黑,不存在白色轮廓");
        }
        else{
            //寻找财报边缘(思路：财报的轮廓面积为最大)
            double maxArea=0;//最大轮廓面积
            for(int i=0;i<contours.size();i++){
                double contourArea=Imgproc.contourArea(contours.get(i));
                if(contourArea>300){
                    resultContours.add(contours.get(i));
                }
            }
        }

        return resultContours;
    }

    public static List<MatOfPoint> getContours3(Mat src){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        List<MatOfPoint> resultContours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();

        //轮廓查找
        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        if(contours.size()<=0){
            logger.info("二值化后图片为全黑,不存在白色轮廓");
        }
        else{
            //寻找财报边缘(思路：财报的轮廓面积为最大)
            double maxArea=0;//最大轮廓面积
            for(int i=0;i<contours.size();i++){
                double contourArea=Imgproc.contourArea(contours.get(i));
                if(contourArea>300){
                    resultContours.add(contours.get(i));
                }
            }
        }

        return resultContours;
    }

    public static Mat removeBackGround3(Mat src,List<MatOfPoint> contours){

        //蒙版
        Mat hole=new Mat(src.size(),CvType.CV_8U,new Scalar(0,0,0));
        Imgproc.drawContours(hole,contours,-1,new Scalar(255,255,255),CV_FILLED);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\hole2.jpg",hole);

        double[] bgColor={255,255,255};

        //去除背景
        for(int i=0;i<src.rows();i++){
            for(int j=0;j<src.cols();j++){
                double[] holePoint=hole.get(i,j);

                if(holePoint[0]==0&&holePoint[0]==0&&holePoint[0]==0)
                    src.put(i,j,bgColor);
            }
        }
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\removeBackGround2.jpg",src);
        return src;
    }

    /**
     * 去除背景
     * @param src 原图
     * @param contour 轮廓
     * @return
     */
    public static Mat removeBackGround(Mat src,MatOfPoint contour){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        contours.add(contour);

        //蒙版
        Mat hole=new Mat(src.size(),CvType.CV_8U,new Scalar(0,0,0));
        Imgproc.drawContours(hole,contours,-1,new Scalar(255,255,255),CV_FILLED);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\hole.jpg",hole);

        //背景颜色替换
//        Point point=contour.toArray()[0];
//        double[] bgColor=src.get((int)point.x,(int)point.y);
        double[] bgColor={255,255,255};

        //去除背景
        for(int i=0;i<src.rows();i++){
            for(int j=0;j<src.cols();j++){
                double[] holePoint=hole.get(i,j);

                if(holePoint[0]==0&&holePoint[0]==0&&holePoint[0]==0)
                    src.put(i,j,bgColor);
            }
        }
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\removeBackGround.jpg",src);
        return src;
    }

    public static Mat operate(Mat operate2,Mat canny){
        double[] black={0};
        double[] white={255};


        double[] cannyPoint=null;
        double[] operate2Point=null;

        for(int i=0;i<canny.rows();i++){
            for(int j=0;j<canny.cols();j++){
                cannyPoint=canny.get(i,j);
                operate2Point=operate2.get(i,j);
                if(cannyPoint[0]==255&&operate2Point[0]==255)
                    operate2.put(i,j,white);
                else
                    operate2.put(i,j,black);
            }
        }
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\operate2-canny.jpg",operate2);
        return operate2;
    }

    public static Mat operate2(Mat adaptiveThreshold,Mat adaptiveThreshold2){
        double[] black={0};
        for(int i=0;i<adaptiveThreshold.rows();i++) {
            for (int j = 0; j < adaptiveThreshold.cols(); j++) {
                if(adaptiveThreshold2.get(i,j)[0]==0)
                    adaptiveThreshold.put(i,j,black);
            }
        }
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\testThreshold3.jpg",adaptiveThreshold);
        return adaptiveThreshold;
    }


    public static void drawContours(Mat src,MatOfPoint contour){
        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        contours.add(contour);
        Imgproc.drawContours(src,contours,-1,new Scalar(0,0,255),3);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\drawContours.jpg",src);
    }

    public static void drawContours2(Mat src,List<MatOfPoint> contours){
        Imgproc.drawContours(src,contours,-1,new Scalar(0,0,255),1);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\drawContours2.jpg",src);
    }

    public static Rect getRect(MatOfPoint contour){
        Rect rect=Imgproc.boundingRect(contour);
        return rect;
    }

    public static void drawRect(Mat src,Rect rect){
        Imgproc.rectangle(src,rect,new Scalar(0,0,255),1);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\drawRect.jpg",src);
    }

    public static Mat cutImg(Mat src,Rect rect){
        Mat src_roi=new Mat(src,rect);
        Mat src_roi_result=new Mat();
        src_roi.copyTo(src_roi_result);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\cutImg.jpg",src_roi_result);
        return src_roi_result;
    }

    /***********************************************************************************/
    public static Mat adaptiveThreshold3(Mat gray){
        Imgproc.adaptiveThreshold(gray, gray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C , Imgproc.THRESH_BINARY, 155, -1);
        Imgcodecs.imwrite("F:\\opencvPhoto\\result\\testThreshold3.jpg",gray);
        return gray;
    }

}
