package com.beawan.xx;

import org.apache.log4j.Logger;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.*;

import static org.opencv.highgui.HighGui.waitKey;

public class FindContours {

    private static final Logger logger=Logger.getLogger(FindContours.class);

    public static String imgPath="F:\\opencvPhoto\\photo\\14.jpg";
    //        public static String imgPath="C:\\Users\\Administrator\\Desktop\\2019412\\7.jpg";
    public static String savePath="F:\\opencvPhoto\\result\\";

    public FindContours(String dllPath){
        System.load(dllPath);
    }

    /**
     * 灰度化
     * @param src 原图
     * @return 灰度图
     */
    public Mat gray(Mat src){
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        return src;
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

    /**
     * 二值化
     * @param src 灰度图
     * @return
     */
    public Mat threshold(Mat src){
        Imgproc.threshold(src,src,175,255,Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
        return src;
    }

    /**
     * 自适应二值化
     * @param src 灰度图
     * @return
     */
    public Mat adaptiveThreshold(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C , Imgproc.THRESH_BINARY, 155, -1);
        Imgproc.medianBlur(src, src, 3);
        return src;
    }

    /**
     * 腐蚀
     * @param src
     * @return
     */
    public Mat erode(Mat src){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Imgproc.erode(src, src, element, new Point(-1, -1), 1);
        return  src;
    }

    /**
     * 开运算
     * @param src
     * @return
     */
    public Mat morphologyEx(Mat src){
        Imgproc.morphologyEx(src,src,Imgproc.MORPH_OPEN,new Mat());
        return src;
    }

    /**
     * 去除背景
     * @param src 原图
     * @param contour 轮廓
     * @return
     */
    public Mat removeBackGround(Mat src,MatOfPoint contour){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        contours.add(contour);

        //蒙版
        Mat hole=new Mat(src.size(),CvType.CV_8U,new Scalar(0,0,0));
        Imgproc.drawContours(hole,contours,-1,new Scalar(255,255,255),-1);
//        Imgcodecs.imwrite(savePath+"hole.jpg",hole);

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
        return src;
    }

    /**
     * 边缘查找
     * @param src 灰度图
     * @return
     */
    public Mat canny(Mat src){
        Imgproc.Canny(src, src,100, 300);
        return src;
    }

    /**
     * 轮廓查找
     * @param src 倾斜矫正后二值化的图片
     * @return 轮廓
     */
    public MatOfPoint getContours(Mat src){

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        MatOfPoint contour=null;
        Mat hierarchy = new Mat();
        //轮廓查找
        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);

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

    /**
     * 消除阴影影响(边缘检测二值化图为白色的像素点位置的二值化图为黑色则填充为白色)
     * @param src 二值化图
     * @param cannyThreshold 边缘检测二值化图
     * @return
     */
    public Mat removeShadow(Mat src,Mat cannyThreshold){
        double[] white={255,255,255};
        for(int i=0;i<src.rows();i++){
            for(int j=0;j<src.cols();j++){
                double[] cannyPoint=cannyThreshold.get(i,j);
                double[] erodePoint=src.get(i,j);
                if(cannyPoint[0]==255&&erodePoint[0]==0){
                    src.put(i,j,white);
                }
            }
        }
        return src;
    }

    /**
     * 预处理(获得原图边缘检测二值化图)
     * @param src 原图
     * @return
     */
    public Mat pretreatment(Mat src){
        Mat gray=gray(src);
        Mat canny=canny(gray);
        Mat dilate=dilate(canny);
        Mat threshold=threshold(dilate);
        return threshold;
    }

    /**
     * 图片裁剪
     * @param src 裁剪图
     * @param rect 裁剪区域
     * @return
     */
    public Mat cutImg(Mat src,Rect rect){
        Mat src_roi=new Mat(src,rect);
        Mat src_roi_result=new Mat();
        src_roi.copyTo(src_roi_result);
        return src_roi_result;
    }

    /**
     * @param src 原图
     * @return
     */
    public MatOfPoint firstStep(Mat src){
        //灰度化
        Mat gray=gray(src.clone());
        //自适应二值化
        Mat threshold=adaptiveThreshold(gray);
        //开运算
        Mat morphologyEx=morphologyEx(threshold);
        //腐蚀
        Mat erode=erode(morphologyEx);
        //得到轮廓
        MatOfPoint contour=getContours(erode);

        return contour;
    }

    /**
     * @param src 根据firstStep函数获得的轮廓裁剪的最小外接正矩形裁剪的图片
     * @return
     */
    public MatOfPoint secondStep(Mat src){
        Mat resultMat=null;
        //预处理
        Mat cannyThreshold=pretreatment(src.clone());
        //灰度化
        Mat gray=gray(src.clone());
        //自适应二值化
        Mat threshold=adaptiveThreshold(gray);
        //腐蚀
        Mat erode=erode(threshold);
        //消除阴影影响
        Mat removeShadow=removeShadow(erode.clone(),cannyThreshold);
        //得到轮廓
        MatOfPoint contour=getContours(removeShadow);

        return contour;
    }


    /**
     * 轮廓检查
     * @param src 原图
     * @param contour 待检查轮廓
     * @param stepName 步骤标记
     * @return
     */
    public boolean checkContour(Mat src,MatOfPoint contour,String stepName){
        logger.info(stepName+"轮廓检查");
        double srcArea=src.width()*src.height();

        if(contour==null){
            logger.info(stepName+"轮廓为空");
            return false;
        }
        else {
            double contourArea=Imgproc.contourArea(contour);
            if(contourArea/srcArea<0.1){
                logger.info(stepName+"轮廓面积不足原图10%");
                return false;
            }

            Rect rect=Imgproc.boundingRect(contour);
            double rectArea=rect.width*rect.height;
            if(contourArea/rectArea<0.5){
                logger.info(stepName+"轮廓面积不足外接矩形50%");
                return false;
            }
        }
        return true;
    }

    /**
     * 得到轮廓
     * @param src 原图
     * @return  noBgCutImg：传给OCR的图片（result为false时不存在）
     */
    public Mat getResultImg(Mat src){

        boolean result=false;//寻找边缘的结果

        MatOfPoint firstStepContour=firstStep(src.clone());

        MatOfPoint secondStepContour=null;

        Rect secondRect=null;

        Mat firstBgCutImg=null;
        Mat noBgCutImg=null;//传给OCR的图片

        Rect rect=null;//最小外接正矩形

        if(firstStepContour!=null) {
            firstBgCutImg = cutImg(src.clone(), Imgproc.boundingRect(firstStepContour));
            secondStepContour=secondStep(firstBgCutImg.clone());

            if(result==false&&checkContour(src.clone(),secondStepContour,"secondStep")){

                //去除背景
                Mat removeBackground=removeBackGround(firstBgCutImg.clone(),secondStepContour);

                //图片裁剪
                secondRect=Imgproc.boundingRect(secondStepContour);
                noBgCutImg=cutImg(removeBackground,secondRect);

                Mat noBgCutImg_gray=gray(noBgCutImg.clone());
                Mat noBgCutImg_Threshold=new Mat();
                Imgproc.adaptiveThreshold(noBgCutImg_gray, noBgCutImg_Threshold, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 25, 5);

                if(checkContour(src.clone(),getContours(noBgCutImg_Threshold),"secondStep->noBgCutImg")){
                    return noBgCutImg;
                }
                else {
                    return null;
                }
            }
            if(result==false&&checkContour(src.clone(),firstStepContour,"firstStap")){

                //去除背景
                Mat removeBackground=removeBackGround(src.clone(),firstStepContour);
                //图片裁剪
                rect=Imgproc.boundingRect(firstStepContour);
                noBgCutImg=cutImg(removeBackground,rect);

                Mat noBgCutImg_gray=gray(noBgCutImg.clone());
                Mat noBgCutImg_Threshold=new Mat();
                Imgproc.adaptiveThreshold(noBgCutImg_gray, noBgCutImg_Threshold, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 25, 5);

                if(checkContour(noBgCutImg.clone(),getContours(noBgCutImg_Threshold),"firstStep->noBgCutImg")){
                    return noBgCutImg;
                }
                else {
                    return null;
                }
            }
            if(result==false) {
                return null;
            }
        }
        else {
            return null;
        }


        return null;
    }

    public static void main(String[] args) {
        FindContours findContours=new FindContours("D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");
        //读入图片
        Mat src = Imgcodecs.imread(FindContours.imgPath);

        /**
         * 调用方式
         */
        Mat result=findContours.getResultImg(src.clone());
        if(result!=null){
            Imgcodecs.imwrite(savePath+"result.jpg",result);
        }
        else
            logger.info("不存在");

    }

}
