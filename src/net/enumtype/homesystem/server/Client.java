package net.enumtype.homesystem.server;

import net.enumtype.homesystem.Main;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;
import org.eclipse.jetty.websocket.api.Session;

import java.io.IOException;
import java.net.InetAddress;
import java.util.List;

public class Client {

    private String name;
    private String password;
    private final ClientManager clientManager;
    private List<String> permissions;
    private Session session;
    private boolean changeConnection;

    public Client(Session session, String name, String password) {
        this.session = session;
        this.name = name;
        this.password = password;
        clientManager = Main.getClientManager();
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

    public void setLoginData(String username, String password) {
        this.name = username;
        this.password = password;
    }

    public void logout() {
        session.close();
        clientManager.removeClient(this);
        Main.getLog().write("User '" + name + "' logged out");
        System.out.print(Main.getCommandPrefix());
    }

    public boolean login() {
        final Log log = Main.getLog();

        if(!session.isSecure()) password = Methods.sha512(password);
        if(clientManager.isDataCorrect(name, password)) {
            log.write("User '" + name + "' logged in with IP " + session.getRemoteAddress().toString());
            System.out.print(Main.getCommandPrefix());
            clientManager.addClient(this);
            return true;
        }else {
            log.write("User tried to login:");
            log.write("USERNAME: " + name);
            log.write("IPADDRESS: " + session.getRemoteAddress().toString());
            System.out.print(Main.getCommandPrefix());
            return false;
        }
    }

    public String getName() {return name;}
    public InetAddress getAddress() {return session.getRemoteAddress().getAddress();}
    public boolean hasPermission(String permission) {return permissions.contains(permission);}
    public boolean isChangingConnection() {return changeConnection;}

}
