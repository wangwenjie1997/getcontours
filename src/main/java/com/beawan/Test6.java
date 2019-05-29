package com.beawan;

import org.apache.log4j.Logger;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class Test6 {

    private static final Logger logger=Logger.getLogger(Test6.class);

    public Test6(String dllPath){
        System.load(dllPath);
    }

    public static String imgPath="F:\\opencvPhoto\\photo\\29.jpg";
    public static String savePath="F:\\opencvPhoto\\result\\";

    public static void main(String[] args) {
        Test6 test=new Test6("D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");

        long startTime = System.currentTimeMillis();

        MatOfPoint2f fourPoint=null;

        //读入图片
        Mat src = Imgcodecs.imread(imgPath);

        Mat gray=test.gray(src.clone());

        Mat canny=test.canny(gray.clone());

        fourPoint=test.getContours(canny);

        if(fourPoint==null){
            Mat adaptiveThreshold=test.adaptiveThreshold2(gray.clone());

            Mat operate=test.operate(gray.clone(),adaptiveThreshold.clone(),canny.clone());

//        Mat adaptiveThreshold2=test.adaptiveThreshold2(operate);
//
//        Mat canny2=test.canny2(adaptiveThreshold2.clone());

            fourPoint=test.getContours(operate);
        }

        long endTime = System.currentTimeMillis();
        logger.info("程序运行时间：" + (endTime - startTime) + "ms");

        if(fourPoint==null)
            logger.info("结论：轮廓不存在");
        else {
            test.drawRect(src.clone(),fourPoint);
        }

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
     * 自适应二值化
     * @param src 灰度图
     * @return
     */
    public Mat adaptiveThreshold(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 25, 5);
        Imgproc.medianBlur(src, src, 3);
//        Mat erode=erode(src);

        Imgcodecs.imwrite(savePath+"threshold.jpg",src);
        return src;
    }

    public Mat adaptiveThreshold2(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 75, -5);
        Imgproc.medianBlur(src, src, 3);

        Imgcodecs.imwrite(savePath+"threshold2.jpg",src);
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
        double heightThresh = 100;//双阀值抑制中的高阀值
        Imgproc.Canny(src, src,lowThresh, heightThresh,3,false);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));
        Imgproc.dilate(src, src, element, new Point(-1, -1), 1);

        Imgcodecs.imwrite(savePath+"canny1.jpg",src);

        return src;
    }

    public Mat canny2(Mat src) {

        //使用Canndy检测边缘
        double lowThresh = 300;//双阀值抑制中的低阀值-100
        double heightThresh = 400;//双阀值抑制中的高阀值
        Imgproc.Canny(src, src,lowThresh, heightThresh,3,false);
//        Imgproc.Canny(blur, cannyMat, lowThresh, heightThresh);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(src, src, element, new Point(-1, -1), 1);

        Imgproc.medianBlur(src, src, 3);

        Imgcodecs.imwrite(savePath+"canny2.jpg",src);

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
        Imgproc.medianBlur(src, src, 5);
        return  src;
    }

    public Mat operate(Mat gray,Mat adaptiveThreshold,Mat canny){
//        Mat threshold=new Mat();
//        Imgproc.threshold(gray, threshold, 175, 255, Imgproc.THRESH_OTSU);

        double[] black={0};
        double[] white={255};
        for(int i=0;i<canny.rows();i++){
            for(int j=0;j<canny.cols();j++){
                if(canny.get(i,j)[0]==255){
                    adaptiveThreshold.put(i, j, white);
                }
            }
        }

        Imgcodecs.imwrite(savePath+"operate.jpg",adaptiveThreshold);
        return adaptiveThreshold;
    }

    public double getEpsilon(MatOfPoint2f curve){
        double arcLength=Imgproc.arcLength(curve,true);
        double epsilon=arcLength*0.005;
//        if(epsilon>30)
//            epsilon=30;
        return epsilon;
    }

    /**
     * 轮廓检测
     * @param src 二值膨胀图
     * @return
     */
    public MatOfPoint2f getContours(Mat src){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        MatOfPoint2f curve=null;

        double maxArea=0;
        double secondArea=0;
        MatOfPoint maxContour=null;
        MatOfPoint secondContour=null;
        double area=0;
        double epsilon=5;
        MatOfPoint2f maxApproxCurve=null;
        MatOfPoint2f secondApproxCurve=null;

        MatOfPoint resultContour=null;
        MatOfPoint2f resultApproxCurve=null;

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
                    secondArea=maxArea;
                    maxArea=area;
                    secondContour=maxContour;
                    maxContour=contour;
                    secondApproxCurve=maxApproxCurve;
                    maxApproxCurve=approxCurve_temp;
                }
            }

        }

        if(maxArea/(src.width()*src.height())<0.3) {
            logger.info("轮廓面积不足");
            resultApproxCurve = null;
        }
        else {
            if(secondArea/maxArea>=0.8){
                logger.info("second");
                resultContour=secondContour;
                resultApproxCurve=secondApproxCurve;
            }
            else {
                resultContour=maxContour;
                resultApproxCurve=maxApproxCurve;
            }
        }

        if(resultContour!=null){
            Mat temp = Imgcodecs.imread(imgPath);
            List<MatOfPoint> list=new ArrayList<MatOfPoint>();
            list.add(resultContour);
            drawContours(temp,list);
        }
        return resultApproxCurve;
    }

    /**************************************画图部分**********************************************/
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

        Mat content=src.clone();
        double[] white={255,255,255};
        for(int i=0;i<src.rows();i++) {
            for (int j = 0; j < src.cols(); j++) {
                if(hole.get(i,j)[0]==0)
                    content.put(i,j,white);
            }
        }
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
