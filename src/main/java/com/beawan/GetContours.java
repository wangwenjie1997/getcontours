package com.beawan;

import org.apache.log4j.Logger;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GetContours {

//    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    private static final Logger logger=Logger.getLogger(GetContours.class);

    private String imgPath;
    private String savePath;

    public GetContours(String imgPath,String savePath,String dllPath){
        this.imgPath=imgPath;
        this.savePath=savePath;
        System.load(dllPath);
    }

    public GetContours(String dllPath){
        System.load(dllPath);
    }

    //读取图片
    public  Mat readImg(String imgPath){
        Mat src=Imgcodecs.imread(imgPath);
        if(src.empty())
            throw new NullPointerException("图片不存在");
        else
            return src;
    }

    //灰度化
    public Mat toGray(Mat src){
        Mat gray=new Mat();
        Imgproc.cvtColor(src,gray,Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur( gray, gray,new Size(3,3));
        Imgcodecs.imwrite(savePath+"gray.jpg",gray);
        return gray;
    }
    //二值化
    public Mat toThreshold(Mat src){
        Mat threshold=new Mat();
        Imgproc.threshold(src,threshold,100,255,Imgproc.THRESH_BINARY);
        Imgcodecs.imwrite(savePath+"threshold.jpg",threshold);
        return threshold;
    }

    /**
     * 轮廓查找
     * @param src 倾斜矫正后二值化的图片
     * @return 若检测到边框返回边框对应的矩形，若检测不到边框返回null
     */
    public Rect findContours(Mat src){
        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);

        Rect paperRect=null;

        if(contours.size()<=0){
            logger.info("二值化后图片为全黑,不存在白色轮廓");
        }
        else{
            //寻找财报边缘(思路：财报的轮廓面积为最大)
            double maxArea=0;
            double imgArea=src.cols()*src.rows();
            MatOfPoint paper=contours.get(0);
            for(int i=0;i<contours.size();i++){
                double contourarea=Imgproc.contourArea(contours.get(i));
                if(contourarea>maxArea){
                    maxArea=contourarea;
                    paper=contours.get(i);
                }
            }

            //如果扫描出来的矩形面积不足原图面积的80%则判定为轮廓不符合
            if(maxArea/imgArea<0.8) {
                logger.info("面积不足原图面积的80%");
            }
            else{//获得外接正矩形
                paperRect=Imgproc.boundingRect(paper);
            }
        }
        return paperRect;
    }

    /**
     * 获取裁剪边框信息
     * 若扫描出符合要求的边框返回Map，否则返回null
     * 边框信息Map内容为：x:矩形左上角x坐标；y:矩形左上角y坐标；width:矩形宽；height:矩形高
     *
     * @param imgPath 图片路径
     * @return
     */
    public Map<String,Integer> getRectMsg(String imgPath){
        long startTime =  System.currentTimeMillis();
        //读入图片
        Mat src= readImg(imgPath);
        //灰度化
        Mat gray=toGray(src);
        //二值化
        Mat threshold=toThreshold(gray);
        //轮廓查找
        Rect paperRect=findContours(threshold);

        Map<String,Integer> map=null;
        //得到矩形信息
        if(paperRect!=null){
            map=new HashMap<String, Integer>();
            map.put("x",paperRect.x);
            map.put("y",paperRect.y);
            map.put("width",paperRect.width);
            map.put("height",paperRect.height);
        }
        long endTime =  System.currentTimeMillis();
        long usedTime = (endTime-startTime)/1000;
        logger.info("获得矩形边框所用时间："+usedTime);
        return map;
    }

    /**
     * 裁剪图片
     *
     * @param imgPath 待裁剪的原图
     * @param x 裁剪矩形左上角x坐标
     * @param y 裁剪矩形左上角y坐标
     * @param width 裁剪矩形宽
     * @param height 裁剪矩形高
     * @param savePath 裁剪出图片的保存路径
     * @return
     */
    public String getImg(String imgPath,int x,int y,int width,int height,String savePath){
        long startTime =  System.currentTimeMillis();
        //读入图片
        Mat src= readImg(imgPath);
        //获得裁剪范围
        Rect paperRect=new Rect(x,y,width,height);
        //裁剪
        Mat paperRect_roi=new Mat(src,paperRect);
        Mat paperRect_roi_result=new Mat();
        paperRect_roi.copyTo(paperRect_roi_result);
        Imgcodecs.imwrite(savePath+"paper.jpg",paperRect_roi_result);
        long endTime =  System.currentTimeMillis();
        long usedTime = (endTime-startTime)/1000;
        logger.info("裁剪边框所用时间："+usedTime);
        return savePath+"paper.jpg";
    }

    public static void main(String[] args) {
        long startTime =  System.currentTimeMillis();

        GetContours getContours=new GetContours("F:\\opencvPhoto\\photo\\yyy.bmp","F:\\opencvPhoto\\result\\","D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");
        Map<String,Integer> map=getContours.getRectMsg(getContours.imgPath);

        long endTime =  System.currentTimeMillis();
        long usedTime = (endTime-startTime)/1000;
        logger.info("所用时间："+usedTime);

        if(map==null)
            logger.info("边界不合符");
        else {
            logger.info(map.get("x"));
            logger.info(map.get("y"));
            logger.info(map.get("width"));
            logger.info(map.get("height"));
            getContours.getImg(getContours.imgPath, map.get("x"), map.get("y"), map.get("width"), map.get("height"), getContours.savePath);
        }
    }

}
