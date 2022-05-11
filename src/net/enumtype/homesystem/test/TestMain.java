package net.enumtype.homesystem.test;

import net.enumtype.homesystem.utils.Methods;

public class TestMain {

    public static final String SALT = "hsnRr4UKMuTsfXHuXQ96NBqJpuJNp49h";

    public static void main(String[] args) {
        System.out.println(Methods.sha512("hallo", SALT));
        TestWebSocket.start();
    }

}