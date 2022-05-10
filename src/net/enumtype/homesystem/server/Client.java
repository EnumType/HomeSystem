package net.enumtype.homesystem.server;

import net.enumtype.homesystem.websocket.WebSocket;
import org.eclipse.jetty.websocket.api.Session;

import java.net.InetAddress;
import java.util.List;

public class Client {

    private final String name;
    private List<String> permissions;
    private Session session;

    public Client(Session session, String name, List<String> permissions) {
        this.session = session;
        this.name = name;
        this.permissions = permissions;
    }

    public void sendMessage(String message) {
        WebSocket.sendCommand(message, session);
    }

    public void sendMessage(String type, List<String> message) {
        for (String s : message) {
            WebSocket.sendCommand("type:" + type + " " + s, session);
        }
    }

    public void setSession(Session session) {
        this.session = session;
    }

    public void updatePermissions(List<String> permissions) {
        this.permissions = permissions;
    }

    public String getName() {
        return name;
    }

    public InetAddress getAddress() {
        return session.getRemoteAddress().getAddress();
    }

    public Session getSession() {
        return session;
    }

    public boolean hasPermission(String permission) {
        return permissions.contains(permission);
    }

}
