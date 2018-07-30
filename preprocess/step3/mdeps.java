package preprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import preprocess.MapDeps;

public class mdeps {
	public static ArrayList<String> dl = new ArrayList<String>();
	public static String GetLoc(String in)
	{
		String r = "-1";
		
		for(int i=0;i<dl.size();i++)
		{
			if(dl.get(i).equals(in))
			{
				r = String.valueOf(i);
				break;
			}
		}
		
		return r;
	}
	public static void main(String[] args)
	{
		int sum = 0;
		try {
			BufferedReader depl = new BufferedReader(new InputStreamReader(new FileInputStream("depsl.txt")));
			String line = "";
	        while((line = depl.readLine()) != null) dl.add(line);
	        System.out.println(dl.size());
			depl.close();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		try {
			BufferedReader mw = new BufferedReader(new InputStreamReader(new FileInputStream(args[0])));
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("mdeps1.txt")));
	        String line = "";
	        while((line = mw.readLine()) != null)
	        {
	        	if(sum++%10000 == 3) System.out.println(sum);
	    		ArrayList<MapDeps> mdl = new ArrayList<MapDeps>();
	        	line = line.replace("), ", ")=");
				line = line.substring(1, line.length()-1);
				String[] lstr = line.split("=");
				for(String str : lstr)
				{
					str = str.replace("(", "-");
					str = str.replace(", ", "-");
					String[] dv = str.split("-");
					if(dv.length==5)
					if(dv[1].matches("^[a-zA-Z]*$") && dv[3].matches("^[a-zA-Z]*$")
							&& dv[1].length() != 0 && dv[3].length() != 0
							&& !dv[1].equals("ROOT") && !dv[3].equals("ROOT"))
					{
						int s1 = 0;
						int s2 = 0;
						for(int i=0;;i++)
						{
							if(i == mdl.size())
							{
								if(s1 == 0)
								{
									mdl.add(new MapDeps(dv[1].toLowerCase()));
									mdl.get(mdl.size()-1).add(dv[3].toLowerCase(), dv[0].toLowerCase());
								}
								if(s2 == 0)
								{
									mdl.add(new MapDeps(dv[3].toLowerCase()));
									mdl.get(mdl.size()-1).add(dv[1].toLowerCase(), "#"+dv[0].toLowerCase());
								}
								break;
							}
							if(s1 == 1 && s2 == 1) break;
							if(mdl.get(i).f.equals(dv[1].toLowerCase()))
							{
								mdl.get(i).add(dv[3].toLowerCase(), dv[0].toLowerCase());
								s1 = 1;
							}
							if(mdl.get(i).f.equals(dv[3].toLowerCase()))
							{
								mdl.get(i).add(dv[1].toLowerCase(), "#"+dv[0].toLowerCase());
								s2 = 1;
							}
						}
					}
				}
				ArrayList<MapDeps> rmdl = new ArrayList<MapDeps>();
				for(int i=0;i<mdl.size();i++) 
				{
					rmdl.add(new MapDeps(mdl.get(i).f));
					for(int j=0;j<mdl.get(i).dml.size();j++)
					{
						String dep = GetLoc(mdl.get(i).dml.get(j).d);
						rmdl.get(i).dml.add(new doubm(mdl.get(i).dml.get(j).w, dep, 1));				
						String s = mdl.get(i).dml.get(j).w;
						for(MapDeps mds:mdl)
						{
							if(s.equals(mds.f))
							{
								for(doubm dbm:mds.dml)
								{
									if(!dbm.w.equals(mdl.get(i).f))
									{
										String sdep = dep+","+GetLoc(dbm.d);
										rmdl.get(i).dml.add(new doubm(dbm.w, sdep, 2));
										String sss = dbm.w;
										for(MapDeps mds3:mdl)
										{
											if(sss.equals(mds3.f))
											{
												for(doubm dbm3:mds3.dml)
												{
													if(!dbm3.w.equals(s) )
													{
														rmdl.get(i).dml.add(new doubm(dbm3.w, sdep+","+GetLoc(dbm3.d), 3));
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
				for(MapDeps md : rmdl)
				{
					out.write(md.f);
					for(int i=0;i<md.dml.size();i++) out.write(" "+md.dml.get(i).w+" "+md.dml.get(i).d);
					out.write("\n");
				}
	        }
	        out.close();
	        mw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
