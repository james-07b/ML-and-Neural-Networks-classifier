package a;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;

public class FileReaderMachine {

    public static void main(String[]args) {


        String pathToCsv = "src/twolves.csv";
        ArrayList<String> list = new ArrayList<String>();
            try {
                System.out.print(",");
                Scanner sc = new Scanner(new File(pathToCsv));
                Scanner in = new Scanner(System.in);
                sc.useDelimiter(",");   //sets the delimiter pattern
                while (sc.hasNext()) {
                    System.out.print(sc.next()+", ");
                    int sentiment = in.nextInt();
                    System.out.print(",");


                }
            }
            catch (Exception e)
            {
                System.out.print("it broke"+e);
            }
    }

}