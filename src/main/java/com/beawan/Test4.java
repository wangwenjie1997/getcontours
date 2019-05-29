package com.beawan;

import org.apache.log4j.Logger;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class Test4 {

    private static final Logger logger=Logger.getLogger(Test4.class);

    public Test4(String dllPath){
        System.load(dllPath);
    }

    public static String imgPath="F:\\opencvPhoto\\photo\\4.jpg";
    public static String savePath="F:\\opencvPhoto\\result\\";

    public static void main(String[] args) {
        Test4 test=new Test4("D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");

        long startTime = System.currentTimeMillis();

        //读入图片
        Mat src = Imgcodecs.imread(imgPath);

        Mat gray=test.gray(src.clone());

        Mat adaptiveThreshold=test.adaptiveThreshold(gray);

        Mat canny=test.canny(adaptiveThreshold.clone());

        Mat operate_src=test.operate(adaptiveThreshold,canny);

        Mat erode=test.erode(operate_src);

        long endTime1 = System.currentTimeMillis();
        logger.info("程序运行时间：" + (endTime1 - startTime) + "ms");

        Mat newSrc=test.getNewSrc(src.clone(),erode);

        Mat newGray=test.gray(newSrc);

        Mat newAdaptiveThreshold=test.adaptiveThreshold2(newGray);

        MatOfPoint contour=test.getContours(newAdaptiveThreshold);

        long endTime = System.currentTimeMillis();
        logger.info("程序运行时间：" + (endTime - startTime) + "ms");

        if(contour==null)
            logger.info("轮廓不存在");
        else {
            List<MatOfPoint> list=new ArrayList<MatOfPoint>();
            list.add(contour);
            test.drawContours(src.clone(),list);
        }

    }

    /**
     * 高斯滤波
     * @param src 原图
     * @return
     */
    public Mat gaussianBlur(Mat src){
        Imgproc.GaussianBlur(src, src, new Size(3,3), 0, 0, Core.BORDER_DEFAULT);
        return src;
    }

    /**
     * 灰度化
     * @param src 原图
     * @return
     */
    public Mat gray(Mat src){
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        return src;
    }

    /**
     * 边缘检测
     * @param src 灰度图
     * @return
     */
    public Mat canny(Mat src) {

        //使用Canndy检测边缘
        double lowThresh = 100;//双阀值抑制中的低阀值-100
        double heightThresh = 300;//双阀值抑制中的高阀值
        Mat cannyMat = new Mat();
        Imgproc.Canny(src, cannyMat,lowThresh, heightThresh,3,false);
//        Imgproc.Canny(blur, cannyMat, lowThresh, heightThresh);

        //膨胀
        Mat dilate = dilate(cannyMat.clone());

        Imgproc.medianBlur(dilate, dilate, 5);

        Imgcodecs.imwrite(savePath+"canny.jpg", dilate);

        return dilate;
    }

    /**
     * 膨胀
     * @param src
     * @return
     */
    public Mat dilate(Mat src){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(src, src, element, new Point(-1, -1), 1);
        return src;
    }

    public Mat operate(Mat adaptiveThreshold,Mat canny){
        double[] black={0};
        for(int i=0;i<canny.rows();i++){
            for(int j=0;j<canny.cols();j++){
                if(canny.get(i,j)[0]==255)
                    adaptiveThreshold.put(i,j,black);
            }
        }
        Imgcodecs.imwrite(savePath+"operate.jpg",adaptiveThreshold);
        return adaptiveThreshold;
    }

    /**
     * 自适应二值化
     * @param src 灰度图
     * @return
     */
    public Mat adaptiveThreshold(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 25, 5);
        Imgproc.medianBlur(src, src, 5);
        Imgcodecs.imwrite(savePath+"threshold.jpg",src);
        return src;
    }

    public Mat adaptiveThreshold2(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C , Imgproc.THRESH_BINARY, 25, -5);
        Imgproc.medianBlur(src, src, 5);
        Imgcodecs.imwrite(savePath+"threshold2.jpg",src);
        return src;
    }

    /**
     * 腐蚀
     * @param src
     * @return
     */
    public Mat erode(Mat src){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5));
        Imgproc.erode(src, src, element, new Point(-1, -1), 1);
        Imgproc.medianBlur(src, src, 3);
        Imgcodecs.imwrite(savePath+"erode.jpg",src);
        return  src;
    }

    public Mat getNewSrc(Mat src,Mat hole){

        Mat newSrc=new Mat();
        src.copyTo(newSrc,hole);
        Imgcodecs.imwrite(savePath+"newSrc.jpg",newSrc);
        return newSrc;
    }

    /**
     * 黑白互换
     * @param src
     * @return
     */
    public static Mat colorInterchange(Mat src){
        Core.bitwise_not(src,src);
        Imgcodecs.imwrite(savePath+"colorInterchange.jpg",src);
        return src;
    }

    public double getEpsilon(MatOfPoint2f curve){
        double arcLength=Imgproc.arcLength(curve,true);
        double epsilon=arcLength*0.005;
        if(epsilon>30)
            epsilon=30;
        return epsilon;
    }

    /**
     * 轮廓检测
     * @param src 二值膨胀图
     * @return
     */
    public MatOfPoint getContours(Mat src){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        MatOfPoint2f curve=null;

        double maxArea=0;
        MatOfPoint maxContour=null;
        double area=0;
        double epsilon=5;
        MatOfPoint2f approxCurve=new MatOfPoint2f();

        //轮廓查找
        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE);

//        logger.info("轮廓数量"+contours.size());

        for(MatOfPoint contour:contours){
            area=Imgproc.contourArea(contour);
            if(area>maxArea){
                curve=new MatOfPoint2f(contour.toArray());

                epsilon=getEpsilon(curve);

                MatOfPoint2f approxCurve_temp=new MatOfPoint2f();
                Imgproc.approxPolyDP(curve,approxCurve_temp,epsilon,true);

                if(approxCurve_temp.total()==4){
                    maxArea=area;
                    maxContour=contour;
                    approxCurve=approxCurve_temp;
                }
            }

        }

        if(maxArea/(src.width()*src.height())<0.3) {
            logger.info("轮廓面积不足");
            maxContour = null;
        }
        else
            drawRect(Imgcodecs.imread(imgPath),approxCurve);

        return maxContour;
    }

    /**
     * 画轮廓
     * @param src 原图
     * @param contours 轮廓
     */
    public void drawContours(Mat src,List<MatOfPoint> contours){

        //蒙版
        Mat hole=new Mat(src.size(),CvType.CV_8U,new Scalar(0,0,0));
        Imgproc.drawContours(hole,contours,-1,new Scalar(255,255,255),-1);
        Imgcodecs.imwrite(savePath+"hole.jpg",hole);

        Mat content=new Mat();
        src.copyTo(content,hole);
        Imgcodecs.imwrite(savePath+"content.jpg",content);

        Imgproc.drawContours(src,contours,-1,new Scalar(0,0,255),3);
        Imgcodecs.imwrite(savePath+"drawContours.jpg",src);
    }

    public void drawRect(Mat src,MatOfPoint2f contour){

        Mat temp=src.clone();

        double[] temp_double=contour.get(0,0);
        Point point1=new Point(temp_double[0],temp_double[1]);

        temp_double=contour.get(1,0);
        Point point2=new Point(temp_double[0],temp_double[1]);

        temp_double=contour.get(2,0);
        Point point3=new Point(temp_double[0],temp_double[1]);

        temp_double=contour.get(3,0);
        Point point4=new Point(temp_double[0],temp_double[1]);

        List<Point> source=new ArrayList<>();
        source.add(point1);
        source.add(point2);
        source.add(point3);
        source.add(point4);

        //对4个点进行排序
        Point centerPoint=new Point(0,0);//质心
        for (Point corner:source){
            centerPoint.x+=corner.x;
            centerPoint.y+=corner.y;
        }
        centerPoint.x=centerPoint.x/source.size();
        centerPoint.y=centerPoint.y/source.size();
        Point leftTop=new Point();
        Point rightTop=new Point();
        Point leftBottom=new Point();
        Point rightBottom=new Point();
        for (int i=0;i<source.size();i++){
            if (source.get(i).x<centerPoint.x&&source.get(i).y<centerPoint.y){
                leftTop=source.get(i);
            }else if (source.get(i).x>centerPoint.x&&source.get(i).y<centerPoint.y){
                rightTop=source.get(i);
            }else if (source.get(i).x<centerPoint.x&& source.get(i).y>centerPoint.y){
                leftBottom=source.get(i);
            }else if (source.get(i).x>centerPoint.x&&source.get(i).y>centerPoint.y){
                rightBottom=source.get(i);
            }
        }

        Scalar color=new Scalar(0,0,255);
        Imgproc.line(temp,leftTop,rightTop,color,5);
        Imgproc.line(temp,rightTop,rightBottom,color,5);
        Imgproc.line(temp,rightBottom,leftBottom,color,5);
        Imgproc.line(temp,leftBottom,leftTop,color,5);

        Imgcodecs.imwrite(savePath+"rect.jpg",temp);

        RotatedRect minRect =Imgproc.minAreaRect(new MatOfPoint2f(contour));
        Point[] points=new Point[4];
        minRect.points(points);
        for (int i = 0; i < 4; i++)
            Imgproc.line(temp, points[i], points[(i+1)%4],new Scalar(0,255,0),3);
        Imgcodecs.imwrite(savePath+"rect.jpg",temp);

    }



}
