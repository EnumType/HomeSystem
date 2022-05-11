package net.enumtype.homesystem.server;

import net.enumtype.homesystem.main.Main;
import org.eclipse.jetty.websocket.api.Session;

import java.io.IOException;
import java.net.InetAddress;
import java.util.List;

public class Client {

    private final String name;
    private List<String> permissions;
    private Session session;
    private boolean changeConnection;

    public Client(Session session, String name, List<String> permissions) {
        this.session = session;
        this.name = name;
        this.permissions = permissions;
    }

    public void sendMessage(String message) {
        try {
            session.getRemote().sendString(message);
        }catch(IOException e) {
            Main.getLog().writeError(e);
        }
    }

    public void sendMessage(String type, List<String> message) {
        for (String s : message) {
            sendMessage("type:" + type + " " + s);
        }
    }

    public void updatePermissions(List<String> permissions) {
        this.permissions = permissions;
    }

    public void changeConnection(boolean changeConnection) {
        this.changeConnection = changeConnection;
    }

    public void setSession(Session session) {
        this.session = session;
    }

    public String getName() {return name;}
    public InetAddress getAddress() {return session.getRemoteAddress().getAddress();}
    public Session getSession() {return session;}
    public boolean hasPermission(String permission) {return permissions.contains(permission);}
    public boolean isChangingConnection() {return changeConnection;}

}
