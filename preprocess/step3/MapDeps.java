package preprocess;

import java.util.ArrayList;

import preprocess.doubm;

public class MapDeps {
	public String f;
	public ArrayList<String> wl;
	public ArrayList<String> dl;
	public ArrayList<doubm> dml;
	public MapDeps(String s){f = s;wl = new ArrayList<String>();dl = new ArrayList<String>();dml=new ArrayList<doubm>();};
	public void add(String s1, String s2){
		wl.add(s1);
		dl.add(s2);
		dml.add(new doubm(s1,s2,0));
	};
}
