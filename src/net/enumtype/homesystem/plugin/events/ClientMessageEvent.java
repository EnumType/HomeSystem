package net.enumtype.homesystem.plugin.events;

import net.enumtype.homesystem.server.Client;
import net.enumtype.homesystem.server.ClientManager;
import net.enumtype.homesystem.server.HomeSystem;
import org.eclipse.jetty.websocket.api.Session;

public class ClientMessageEvent implements Event {

    private final Client client;
    private final Session session;
    private final String message;

    public ClientMessageEvent(Session session, String message) {
        final ClientManager clientManager = HomeSystem.getClientManager();
        this.client = clientManager.isLoggedIn(session) ? clientManager.getClient(session) : null;
        this.session = session;
        this.message = message;
    }

    public Client getClient() {
        return client;
    }

    public Session getSession() {
        return session;
    }

    public String getMessage() {
        return message;
    }

    public boolean isLoggedIn() {return client != null;}

}