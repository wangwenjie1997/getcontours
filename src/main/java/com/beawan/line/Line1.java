package com.beawan.line;

import org.apache.log4j.Logger;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.LineSegmentDetector;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.LSD_REFINE_STD;


/**
 * 参考链接:https://blog.csdn.net/yomo127/article/details/52045146
 */
public class Line1 {

    private static final Logger logger=Logger.getLogger(Line1.class);

    public Line1(String dllPath){
        System.load(dllPath);
    }

    public static String imgPath="F:\\opencvPhoto\\photo\\t.bmp";
    public static String savePath="F:\\opencvPhoto\\result\\";

    public static void main(String[] args) {
        Line1 line=new Line1("D:\\Program Files\\opencv\\build\\java\\x64\\opencv_java401.dll");

        long startTime = System.currentTimeMillis();
        long endTime=0;

        //读入图片
        Mat src = Imgcodecs.imread(imgPath);

        Mat gray=line.gray(src.clone());

//        Mat adaptiveThreshold=line.adaptiveThreshold(gray.clone());

//        Mat morphologyEx=line.morphologyEx(adaptiveThreshold);

        Mat adaptiveThreshold=new Mat();

        Imgproc.threshold(gray, adaptiveThreshold, 100, 255, Imgproc.THRESH_BINARY_INV);

        Imgcodecs.imwrite(savePath+"threshold.jpg",adaptiveThreshold);

//        Mat colorInterchange=line.colorInterchange(adaptiveThreshold.clone());

        /**版本一**/
        Mat horizontal=line.horizontal(adaptiveThreshold.clone());
        Mat vertical=line.vertical(adaptiveThreshold.clone());
//        Mat operate=line.operate(horizontal.clone(),vertical.clone(),adaptiveThreshold.clone());

        Mat getOr=line.getOr(horizontal.clone(),vertical.clone());
        Mat getAnd=line.getAnd(horizontal.clone(),vertical.clone());
        Mat dilate=line.dilate(getAnd.clone());
        List<Point> points=line.getPoints(dilate.clone(),src.clone());



        Mat temp=src.clone();
        for(Point p:points)
            Imgproc.circle(temp,p,5,new Scalar(0,0,255),-1);
        Imgcodecs.imwrite(savePath+"temp.jpg",temp);


        /**版本二**/
//        Mat horizontal=line.horizontal(colorInterchange.clone());
//        Mat operate=line.operate(horizontal,adaptiveThreshold);
//        Mat colorInterchange2=line.colorInterchange(operate.clone());
//        Mat houghLinesP=line.houghLinesP(horizontal.clone(),colorInterchange2.clone());


        /**版本三**/
//        Mat houghLinesP2=line.houghLinesP2(dilate.clone(),adaptiveThreshold.clone());

        /**想法一**/
//        Mat dilate=line.dilate(adaptiveThreshold);
//        line.test(adaptiveThreshold.clone());


//        double[] white={255,255,255};
//        for(int i=0;i<src.rows();i++){
//            for(int j=0;j<src.cols();j++){
//                if(houghLinesP2.get(i,j)[0]==255)
//                    src.put(i,j,white);
//            }
//        }
//        Imgcodecs.imwrite(savePath+"result.jpg",src);

        logger.info("程序运行时间：" + (endTime - startTime) + "ms");
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
     * 腐蚀
     * @param src
     * @return
     */
    public Mat erode(Mat src){
//        int erodeSize = src.cols() / 200;
//        if (erodeSize % 2 == 0)
//            erodeSize++;
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Imgproc.erode(src, src, element, new Point(-1, -1), 1);
        Imgcodecs.imwrite(savePath+"erode.jpg",src);
        return  src;
    }

    /**
     * 膨胀
     * @param src
     * @return
     */
    public Mat dilate(Mat src){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Imgproc.dilate(src, src, element, new Point(-1, -1), 1);
        Imgcodecs.imwrite(savePath+"dilate.jpg",src);
        return src;
    }

    public Mat morphologyEx(Mat src){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1,1));
        Imgproc.morphologyEx(src,src,Imgproc.MORPH_OPEN,element);
        Imgcodecs.imwrite(savePath+"morphologyEx.jpg",src);
        return src;
    }

    /**
     * 高斯模糊
     * @param src
     * @return
     */
    public Mat gaussianBlur(Mat src){
        int blurSize = src.cols() / 200;
        if (blurSize % 2 == 0)
            blurSize++;
        Imgproc.GaussianBlur(src, src,new Size(blurSize, blurSize), 0, 0);
        return src;
    }

    /**
     * 自适应二值化
     * @param src 灰度图
     * @return
     */
    public Mat adaptiveThreshold(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 25, 7);
        Imgproc.medianBlur(src, src, 5);
        Imgcodecs.imwrite(savePath+"adaptiveThreshold.jpg",src);
        return src;
    }

    public Mat adaptiveThreshold2(Mat src){
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, -5);
//        Imgproc.medianBlur(src, src, 3);
//        Imgcodecs.imwrite(savePath+"threshold.jpg",src);
        Imgcodecs.imwrite(savePath+"adaptiveThreshold2.jpg",src);
        return src;
    }

    /**
     * 黑白互换
     * @param src
     * @return
     */
    public Mat colorInterchange(Mat src){
        Core.bitwise_not(src,src);
        Imgcodecs.imwrite(savePath+"colorInterchange.jpg",src);
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

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(src, src, element, new Point(-1, -1), 1);

        Imgcodecs.imwrite(savePath+"canny.jpg",src);

        return src;
    }

    public void createLineSegmentDetector(Mat src,Mat adaptiveThreshold){
        LineSegmentDetector lineSegmentDetector = Imgproc.createLineSegmentDetector(LSD_REFINE_STD);
        Mat lines=new Mat();
        lineSegmentDetector.detect(adaptiveThreshold,lines);

        List<MatOfPoint> list=new ArrayList<MatOfPoint>();
        Scalar scalar=new Scalar(255);

        for (int x = 0; x < lines.rows(); x++) {
            double[] vec = lines.get(x, 0);
            double x1 = vec[0], y1 = vec[1], x2 = vec[2], y2 = vec[3];
            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);
//            double len=Math.sqrt(Math.pow(start.x-end.x,2)+Math.pow(start.y-end.y,2));
//            if(len>100)
                Imgproc.line(src, start, end, scalar, 3);
        }


        Imgcodecs.imwrite(savePath+"lsd.jpg",src);
    }

    public Mat houghLinesP(Mat horizontal,Mat adaptiveThreshold){
        double angle;
        Mat lines=new Mat();
        Imgproc.HoughLinesP(adaptiveThreshold, lines, 1, Math.PI / 180, 50, 0, 10);
        Mat hole=new Mat(adaptiveThreshold.size(),CvType.CV_8U,new Scalar(0,0,0));
        for (int x = 0; x < lines.rows(); x++) {
            double[] vec = lines.get(x, 0);
            double x1 = vec[0], y1 = vec[1], x2 = vec[2], y2 = vec[3];
            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);
            Imgproc.line(hole, start, end, new Scalar(255, 255, 255), 5);
        }
        Imgcodecs.imwrite(savePath+"houghLinesP-2.jpg",hole);

        Imgproc.HoughLinesP(adaptiveThreshold, lines, 1, Math.PI / 180, 50, 0, 10);
        for (int x = 0; x < lines.rows(); x++) {
            double[] vec = lines.get(x, 0);
            double x1 = vec[0], y1 = vec[1], x2 = vec[2], y2 = vec[3];
            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);
            angle=180/Math.PI*Math.atan(Math.abs(y1-y2)/Math.abs(x1-x2));
            if(angle>80)
                Imgproc.line(horizontal, start, end, new Scalar(255, 255, 255), 3);
        }

        Imgcodecs.imwrite(savePath+"houghLinesP.jpg",horizontal);


        return horizontal;
    }

    public Mat houghLinesP2(Mat horizontal,Mat adaptiveThreshold){
        Mat lines=new Mat();
        Imgproc.HoughLinesP(adaptiveThreshold, lines, 1, Math.PI / 180, 50, 0, 0);
        Mat hole=new Mat(adaptiveThreshold.size(),CvType.CV_8U,new Scalar(0,0,0));

        double angle,len;
        for (int x = 0; x < lines.rows(); x++) {
            double[] vec = lines.get(x, 0);
            double x1 = vec[0], y1 = vec[1], x2 = vec[2], y2 = vec[3];
            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);
//            len=Math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
            angle=180/Math.PI*Math.atan(Math.abs(y1-y2)/Math.abs(x1-x2));
            if(angle>80||angle<10)
                Imgproc.line(hole, start, end, new Scalar(255, 255, 255), 3);

        }
        Imgcodecs.imwrite(savePath+"houghLinesP2.jpg",hole);
        return hole;
    }

    public void createLineSegmentDetector(Mat adaptiveThreshold){
        LineSegmentDetector lineSegmentDetector = Imgproc.createLineSegmentDetector(LSD_REFINE_STD);
        Mat lines=new Mat();
        lineSegmentDetector.detect(adaptiveThreshold,lines);

        List<MatOfPoint> list=new ArrayList<MatOfPoint>();
        Scalar scalar=new Scalar(255);
        Mat hole=new Mat(adaptiveThreshold.size(),CvType.CV_8U,new Scalar(0,0,0));
        for (int x = 0; x < lines.rows(); x++)
        {
            double[] vec = lines.get(x, 0);
            double x1 = vec[0], y1 = vec[1], x2 = vec[2], y2 = vec[3];
            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);
            double len=Math.sqrt(Math.pow(start.x-end.x,2)+Math.pow(start.y-end.y,2));
            if(len>100)
                Imgproc.line(hole, start, end, scalar, 3);
        }


        Imgcodecs.imwrite(savePath+"lsd.jpg",hole);
    }

    /****************************************************************************/
    public Mat horizontal(Mat adaptiveThreshold){
        int scale = 30; //这个值越大，检测到的直线越多
        int horizontalsize = adaptiveThreshold.cols() / scale;
        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(horizontalsize, 1));
        // 先腐蚀再膨胀
        Imgproc.erode(adaptiveThreshold, adaptiveThreshold, horizontalStructure,new Point(-1, -1));
        Imgproc.dilate(adaptiveThreshold, adaptiveThreshold, horizontalStructure,new Point(-1, -1));

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(adaptiveThreshold, adaptiveThreshold, element, new Point(-1, -1), 1);
        Imgcodecs.imwrite(savePath+"horizontal.jpg",adaptiveThreshold);
        return adaptiveThreshold;
    }

    public Mat vertical(Mat adaptiveThreshold){
        int scale = 30; //这个值越大，检测到的直线越多
        int verticalsize  = adaptiveThreshold.rows() / scale;
        Mat verticalStructure  = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(1, verticalsize));
        // 先腐蚀再膨胀
        Imgproc.erode(adaptiveThreshold, adaptiveThreshold, verticalStructure,new Point(-1, -1));
        Imgproc.dilate(adaptiveThreshold, adaptiveThreshold, verticalStructure,new Point(-1, -1));
        Imgcodecs.imwrite(savePath+"vertical.jpg",adaptiveThreshold);
        return adaptiveThreshold;
    }

    public Mat getAnd(Mat horizontal,Mat vertical){
        Mat and=new Mat();
        Core.bitwise_and(horizontal,vertical,and);
        Imgcodecs.imwrite(savePath+"and.jpg",and);
        return and;
    }

    public Mat getOr(Mat horizontal,Mat vertical){
        Mat or=new Mat();
        Core.bitwise_or(horizontal,vertical,or);
        Imgcodecs.imwrite(savePath+"or.jpg",or);
        return or;
    }

    public Mat operate(Mat horizontal,Mat vertical,Mat adaptiveThreshold){
        double[] black={0};
        for(int i=0;i<adaptiveThreshold.rows();i++){
            for(int j=0;j<adaptiveThreshold.cols();j++){
                if(horizontal.get(i,j)[0]==255){
                    adaptiveThreshold.put(i, j, black);
                }
            }
        }

        Imgcodecs.imwrite(savePath+"operate_step.jpg",adaptiveThreshold);

        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(adaptiveThreshold,contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
        RotatedRect rotatedRect =null;
        double width,height,angel,minLen,minY,maxY,minX,maxX;
        List<MatOfPoint> drawContoues=new ArrayList<MatOfPoint>();
        boolean flag;
        Point[] points=new Point[4];
        for(MatOfPoint contour:contours){
            minY=vertical.height()+1;
            maxY=-1;
            minX=vertical.width()+1;
            maxX=-1;
            flag=false;
            rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
            width=rotatedRect.size.width;
            height=rotatedRect.size.height;
            angel=rotatedRect.angle;
//            minLen=width<height?width:height;

            rotatedRect.points(points);

            if(angel>-10||angel<-80) {

                for(Point p:points){
                    if(p.y>maxY)
                        maxY=p.y;
                    if(p.y<minY)
                        minY=p.y;
                    if(p.x>maxX)
                        maxX=p.x;
                    if(p.x<minX)
                        minX=p.x;
                }
                if((maxY-minY)>(maxX-minX)&&(width/height>3||height/width>3)) {
                    logger.info(angel);
                    flag = true;
                }
            }

            if(flag)
                drawContoues.add(contour);

        }

        Imgproc.drawContours(vertical,drawContoues,-1,new Scalar(255,255,255),3);

        Imgcodecs.imwrite(savePath+"operate.jpg",vertical);

        return vertical;
    }

    public List<Point> getPoints(Mat getAnd,Mat src){
        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        List<Point> points=new ArrayList<Point>();
        //轮廓查找
        Imgproc.findContours(getAnd,contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
        RotatedRect rotatedRect =null;
        for(MatOfPoint contour:contours){
            rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
            points.add(rotatedRect.center);
        }
        return points;
    }

    public void houghLines(Mat src,Mat adaptiveThreshold){

        Mat lines=new Mat();
        Imgproc.HoughLines(adaptiveThreshold,lines,1,Math.PI/180,100,0,0);

        Scalar scalar=new Scalar(255);

        for (int x = 0; x < lines.rows(); x++)
        {
            double[] vec = lines.get(x, 0);

            double rho = vec[0];
            double theta = vec[1];
            double angle=180/Math.PI*theta%90;

            Point pt1 = new Point();
            Point pt2 = new Point();

            double a = Math.cos(theta);
            double b = Math.sin(theta);

            double x0 = a * rho;
            double y0 = b * rho;

            pt1.x = Math.round(x0 + 1000 * (-b));
            pt1.y = Math.round(y0 + 1000 * (a));
            pt2.x = Math.round(x0 - 1000 * (-b));
            pt2.y = Math.round(y0 - 1000 * (a));

            if (angle >80) {
                Imgproc.line(src, pt1, pt2, new Scalar(0, 0, 255), 1);
            }

        }
        Imgcodecs.imwrite(savePath+"houghLines.jpg",src);
    }

    public Mat test(Mat adaptiveThreshold){
        Mat hole=new Mat(adaptiveThreshold.size(),CvType.CV_8U,new Scalar(0,0,0));
        Mat hole1=hole.clone();
        Mat hole2=hole.clone();
        Mat hole3=hole.clone();
        Mat hole4=hole.clone();

        Mat temp=adaptiveThreshold.clone();

        int num=0;
        double[] black={0};
        double[] color;
        boolean flag=false;
        for(int i=0;i<adaptiveThreshold.rows();i++){
            num=0;
            for(int j=0;j<adaptiveThreshold.cols();j++){
                color=adaptiveThreshold.get(i,j);
                if(color[0]==255)
                    flag=true;
                else{
                    flag=false;
                    num=0;
                }
                if(num>1)
                    continue;
                if(flag&&num<=1){
                    hole1.put(i,j,color);
                    hole.put(i,j,color);
                    temp.put(i,j,black);
                    num++;
                }

//                System.out.println(num);
            }
        }



        for(int i=0;i<adaptiveThreshold.rows();i++) {
            num=0;
            for(int j=adaptiveThreshold.cols()-1;j>=0;j--){
                color=adaptiveThreshold.get(i,j);
                if(color[0]==255)
                    flag=true;
                else{
                    flag=false;
                    num=0;
                }
                if(num>1)
                    continue;
                if(flag&&num<=1){
                    hole2.put(i,j,color);
                    num++;
                }
            }
        }


        for(int i=0;i<adaptiveThreshold.cols();i++){
            num=0;
            for(int j=0;j<adaptiveThreshold.rows();j++){
                color=adaptiveThreshold.get(j,i);
                if(color[0]==255)
                    flag=true;
                else{
                    flag=false;
                    num=0;
                }
                if(num>1)
                    continue;
                if(flag&&num<=1){
                    hole3.put(j,i,color);

                    num++;
                }
            }
        }



        for(int i=0;i<adaptiveThreshold.cols();i++) {
            num = 0;
            for(int j=adaptiveThreshold.rows()-1;j>=0;j--){
                color=adaptiveThreshold.get(j,i);
                if(color[0]==255)
                    flag=true;
                else{
                    flag=false;
                    num=0;
                }
                if(num>1)
                    continue;
                if(flag&&num<=1){
                    hole4.put(j,i,color);
                    hole.put(j,i,color);
                    temp.put(j,i,black);
                    num++;
                }
            }
        }

        Imgcodecs.imwrite(savePath+"hole.jpg",hole);
        Imgcodecs.imwrite(savePath+"hole1.jpg",hole1);
        Imgcodecs.imwrite(savePath+"hole2.jpg",hole2);
        Imgcodecs.imwrite(savePath+"hole3.jpg",hole3);
        Imgcodecs.imwrite(savePath+"hole4.jpg",hole4);

        Imgcodecs.imwrite(savePath+"temp.jpg",temp);

        return null;
    }

}
