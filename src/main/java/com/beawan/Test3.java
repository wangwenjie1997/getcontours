package com.beawan;

import org.apache.log4j.Logger;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgproc.CV_FILLED;

public class Test3 {

    private static final Logger logger=Logger.getLogger(Test3.class);

    public Test3(String dllPath){
        System.load(dllPath);
    }

    public static String imgPath="F:\\opencvPhoto\\photo\\52.jpg";
    public static String savePath="F:\\opencvPhoto\\result\\";

    public static void main(String[] args) {
        Test3 test=new Test3("D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");

        //读入图片
        Mat src = Imgcodecs.imread(imgPath);

        Mat canny=test.canny(src.clone());

        Mat src_temp=test.operate(src.clone(),canny.clone());

        Mat gray=test.gray(src_temp.clone());

        Mat adaptiveThreshold=test.adaptiveThreshold(gray.clone());

        MatOfPoint contour=test.getContours(adaptiveThreshold.clone());

        if(contour==null)
            logger.info("轮廓不存在");
        else {
            List<MatOfPoint> list=new ArrayList<MatOfPoint>();
            list.add(contour);
            test.drawContours(src.clone(),list);
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
     * 边缘检测
     * @param src 灰度图
     * @return
     */
    public Mat canny(Mat src) {
        //平滑处理
//        Mat blur = new Mat();
//        Imgproc.blur(src, blur, new Size(2, 2));

        //轮廓
        //使用Canndy检测边缘
        double lowThresh = 200;//双阀值抑制中的低阀值-100
        double heightThresh = 300;//双阀值抑制中的高阀值
        Mat cannyMat = new Mat();
        Imgproc.Canny(src, cannyMat,lowThresh, heightThresh,3,false);
//        Imgproc.Canny(blur, cannyMat, lowThresh, heightThresh);

        //膨胀
        Mat dilate = new Mat();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(cannyMat, dilate, element, new Point(-1, -1), 3);

        Imgproc.medianBlur(dilate, dilate, 5);

//        Imgproc.threshold(dilate, dilate, 175, 255, Imgproc.THRESH_BINARY);

        Imgcodecs.imwrite(savePath+"canny.jpg", dilate);

        return dilate;
    }

    public Mat gaussianBlur(Mat src){
        Imgproc.GaussianBlur(src, src, new Size(3,3), 0, 0, Core.BORDER_DEFAULT);
        Imgcodecs.imwrite(savePath+"gaussianBlur.jpg",src);
        return src;
    }

    /**
     * 自适应二值化
     * @param src 灰度图
     * @return
     */
    public Mat adaptiveThreshold(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C , Imgproc.THRESH_BINARY, 77, -5);
        Imgproc.medianBlur(src, src, 5);

        src=erode(src.clone());

        Imgcodecs.imwrite(savePath+"adaptiveThreshold.jpg",src);
        return src;
    }

    /**
     * 膨胀
     * @param src
     * @return
     */
    public Mat dilate(Mat src){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));
        Imgproc.dilate(src, src, element, new Point(-1, -1), 1);
        Imgcodecs.imwrite(savePath+"dilate.jpg",src);
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
        Imgcodecs.imwrite(savePath+"erode.jpg",src);
        return  src;
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
        double arcLength=0;
        MatOfPoint2f approxCurve=new MatOfPoint2f();

        //轮廓查找
        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE);

//        logger.info("轮廓数量"+contours.size());

        for(MatOfPoint contour:contours){
            area=Imgproc.contourArea(contour);
            if(area>maxArea){
                curve=new MatOfPoint2f(contour.toArray());
                arcLength=Imgproc.arcLength(curve,true);

                epsilon=arcLength*0.01;
//                if(epsilon<10)
//                    epsilon=10;
//                else if(epsilon>30)
//                    epsilon=30;

                MatOfPoint2f approxCurve_temp=new MatOfPoint2f();
                Imgproc.approxPolyDP(curve,approxCurve_temp,epsilon,true);

                if(approxCurve_temp.total()==4){
                    maxArea=area;
                    maxContour=contour;
                    approxCurve=approxCurve_temp;
                }
            }

        }

        if(maxArea/(src.width()*src.height())<0.3)
            maxContour=null;
        else
            drawRect(Imgcodecs.imread(imgPath),approxCurve);

        return maxContour;
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

    public Mat operate(Mat src,Mat canny){
        double[] black={0,0,0};
        for(int i=0;i<canny.rows();i++){
            for(int j=0;j<canny.cols();j++){
                if(canny.get(i,j)[0]==255)
                    src.put(i,j,black);
            }
        }
        Imgcodecs.imwrite(savePath+"operate.jpg",src);
        return src;
    }

    /**
     * 画轮廓
     * @param src 原图
     * @param contours 轮廓
     */
    public void drawContours(Mat src,List<MatOfPoint> contours){

//        List<MatOfPoint> list=new ArrayList<MatOfPoint>();
//        Scalar scalar=null;
//        double r,g,b;
//        for(MatOfPoint contour:contours){
//            if(Imgproc.contourArea(contour)<300)
//                continue;
//            list.add(contour);
//            r=Math.random()*256;
//            g=Math.random()*256;
//            b=Math.random()*256;
//            scalar=new Scalar(r,g,b);
//            Imgproc.drawContours(src,list,-1,scalar,3);
//            list.clear();
//
//        }

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
        Imgproc.line(temp,leftTop,rightTop,color,3);
        Imgproc.line(temp,rightTop,rightBottom,color,3);
        Imgproc.line(temp,rightBottom,leftBottom,color,3);
        Imgproc.line(temp,leftBottom,leftTop,color,3);

        Imgcodecs.imwrite(savePath+"rect.jpg",temp);

    }

}
