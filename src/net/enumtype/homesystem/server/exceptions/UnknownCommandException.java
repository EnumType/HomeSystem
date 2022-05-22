package net.enumtype.homesystem.server.exceptions;

import net.enumtype.homesystem.server.Client;

public class UnknownCommandException extends Exception {

    public UnknownCommandException(String command) {
        super("Unknown command '" + command + "'");
    }

    public UnknownCommandException(Client client, String command) {
        super("Unknown command '" + command + "' issued by '" + client.getName() + "' with address " + client.getAddress());
    }

}