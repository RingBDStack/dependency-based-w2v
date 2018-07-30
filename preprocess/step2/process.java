import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;

import java.io.*;
import java.util.Collection;

public class process {
    public static int num = 0;
    public static void main(String[] args) throws IOException {
        try {
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[0]+".parse")));
            String modelpath = "edu/stanford/nlp/models/lexparser/englishFactored.ser.gz";
            LexicalizedParser lp = LexicalizedParser.loadModel(modelpath);
            BufferedReader mw = new BufferedReader(new InputStreamReader(new FileInputStream(args[0])));
            String line = "";
            while((line = mw.readLine()) != null)
            {
                if (line.contains(".") || line.contains("?") || line.contains("!")) {
                    line = line.replaceAll("\\*|#|=|\\|(|)|&|^|%|$|@|~|`|}|\\{|]|\\[|>|<", "");
                    String[] str = line.split("\\.|\\?|!");
                    for (String s : str) {
                        if (s.split(" ").length > 10) {
                            while (true) {
                                if (s.length() == 1) {
                                    s = "";
                                    break;
                                }
                                char c = s.charAt(0);
                                if (c < 65 || c > 90) {
                                    s = s.substring(1, s.length());
                                } else break;
                            }
                            if (s.split(" ").length > 10) {
                                Tree t = lp.parse(s);
                                EnglishGrammaticalStructure es = new EnglishGrammaticalStructure(t);
                                Collection<TypedDependency> tdl = es.typedDependenciesCollapsed();
                                if(num%1000 == 0) System.out.println("out: " + num);
                                out.write(tdl.toString() + "\n");
                                out.flush();
                                num++;
                            }
                        }
                    }
                }
            }
            out.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }
}
