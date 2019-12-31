package net.javaexception.homesystem.utils;

import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.X509EncodedKeySpec;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;

public class Crypto {
	
	private String algo;
	private int lenght;
	private PublicKey pubkey;
	private PrivateKey privkey;
	
	public Crypto(String algo, int lenght) {
		this.algo = algo;
		this.lenght = lenght;
	}
	
	public void generate() {
		try {
			Log.write(Methods.createPrefix() + "Generating " + lenght + "bit RSA-Keys...", true);
			final KeyPairGenerator generator = KeyPairGenerator.getInstance(algo);
			generator.initialize(lenght);
			
			KeyPair key = generator.generateKeyPair();
			
			pubkey = key.getPublic();
			privkey = key.getPrivate();
			
			System.out.println("------------------------------------------------------------------------------------------------------------------------------------------------------------");
			System.out.println(pubkey);
			System.out.println("------------------------------------------------------------------------------------------------------------------------------------------------------------");
			Log.write(Methods.createPrefix() + "Generated Keys", true);
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Crypto(47): " + e.getMessage(), false);
		}
	}
	
	public byte[] encrypt(String input, PublicKey key) {
		byte[] out = null;
				
		Cipher cipher;
		try {
			cipher = Cipher.getInstance("RSA/ECB/PKCS1Padding");
			cipher.init(Cipher.ENCRYPT_MODE, key);
			out = cipher.doFinal(input.getBytes());
		} catch (NoSuchAlgorithmException | NoSuchPaddingException |
				InvalidKeyException | IllegalBlockSizeException | BadPaddingException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Crypto(62): " + e.getMessage(), false);
		}
		
		return out;
	}
	
	public String decrypt(byte[] input, PrivateKey key) {
		byte[] out = null;
		
		Cipher cipher;
		try {
			cipher = Cipher.getInstance("RSA/ECB/PKCS1Padding");
			cipher.init(Cipher.DECRYPT_MODE, key);
			out = cipher.doFinal(input);
		} catch (NoSuchAlgorithmException | NoSuchPaddingException |
				InvalidKeyException | IllegalBlockSizeException | BadPaddingException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Crypto(79): " + e.getMessage(), false);
		}
		
		return new String(out);
	}
	
	public PublicKey readPublicKey(byte[] pubkey) {
		PublicKey key = null;
		
		try {
			X509EncodedKeySpec spec = new X509EncodedKeySpec(pubkey);
			KeyFactory factory = KeyFactory.getInstance(algo);
			key = factory.generatePublic(spec);
		} catch (NoSuchAlgorithmException | InvalidKeySpecException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Crypto(94): " + e.getMessage(), false);
		}
		
		return key;
	}
	
	public PublicKey getPublicKey() {
		return pubkey;
	}
	
	public PrivateKey getPrivateKey() {
		return privkey;
	}
	
	public int getLenght() {
		return lenght;
	}
	
}
