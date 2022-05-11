package net.enumtype.homesystem.utils;

import net.enumtype.homesystem.server.Client;

public class UnknownCommandException extends Exception {

    public UnknownCommandException(Client client, String command) {
        super("Unknown command '" + command + " issued by '" + client.getName() + "' with address " + client.getAddress());
    }

}
