package net.enumtype.homesystem.plugin.events;

import net.enumtype.homesystem.server.Client;
import org.eclipse.jetty.websocket.api.Session;

import java.net.InetAddress;

public class ClientLogoutEvent implements Event {

    private final Client client;

    public ClientLogoutEvent(Client client) {
        this.client = client;
    }

    public Client getClient() {
        return client;
    }

    public InetAddress getAddress() {
        return client.getAddress();
    }

    public Session getSession() {
        return client.getSession();
    }

}
