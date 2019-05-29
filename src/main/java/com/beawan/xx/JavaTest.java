package com.beawan.xx;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

public class JavaTest {

    public static void main(String[] args) throws Exception {

        String str="abc";
        System.out.println("修改前："+str);
        Field valueField=str.getClass().getDeclaredField("value");
        valueField.setAccessible(true);
        char[] temp=(char[])valueField.get(str);
        temp[0]='1';
        temp[1]='2';
        temp[2]='3';
        System.out.println("修改后："+str);

    }

}
