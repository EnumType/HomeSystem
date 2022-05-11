package net.enumtype.homesystem.server;

public class UnknownCommandException extends Exception {

    public UnknownCommandException(String error) {
        super(error);
    }

    public UnknownCommandException(Client client, String command) {
        super("Unknown command '" + command + " issued by '" + client.getName() + "' with address " + client.getAddress());
    }

}
