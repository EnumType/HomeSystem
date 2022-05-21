package net.enumtype.homesystem.plugin.events;

import org.eclipse.jetty.websocket.api.Session;

public class ClientConnectEvent implements Event {

    private final Session session;

    public ClientConnectEvent(Session session) {
        this.session = session;
    }

    public Session getSession() {
        return session;
    }

}