package net.javaexception.homesystem.server;

import net.javaexception.homesystem.websocket.WebSocket;

import java.net.InetAddress;
import java.util.List;

public class Client {

    private final String name;
    private final InetAddress address;
    private List<String> permissions;

    public Client(InetAddress address, String name, List<String> permissions) {
        this.address = address;
        this.name = name;
        this.permissions = permissions;
    }

    public void sendMessage(String message) {
        WebSocket.sendCommand(message, address);
    }

    public void sendMessage(String type, List<String> message) {
        for(int i = 0; i < message.size(); i++) {
            WebSocket.sendCommand("type:" + type + " " + message.get(i), address);
        }
    }

    public void updatePermissions(List<String> permissions) {
        this.permissions = permissions;
    }

    public String getName() {
        return name;
    }

    public InetAddress getAddress() {
        return address;
    }

    public boolean hasPermission(String permission) {
        return permissions.contains(permission);
    }

}
