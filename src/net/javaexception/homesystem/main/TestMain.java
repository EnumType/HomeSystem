package net.javaexception.homesystem.main;

import net.javaexception.homesystem.xmlrpc.RoomManager;

public class TestMain {

    public static void main(String[] args) {
        final RoomManager rm = new RoomManager();
        rm.load();
    }

}
