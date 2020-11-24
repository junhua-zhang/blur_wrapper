using System;
using System.IO;
using Darknet;
using static Darknet.YoloWrapper;

namespace YoloTest_CS
{
    class Program
    {
        YoloWrapper yolodetector;
        public static byte[] GetPictureData(string imagePath)
        {
            FileStream fs = new FileStream(imagePath, FileMode.Open);
            byte[] byteData = new byte[fs.Length];
            fs.Read(byteData, 0, byteData.Length);
            fs.Close();
            return byteData;
        }

        static void Main(string[] args)
        {
            YoloWrapper yolodetector;
            try
            {
                yolodetector = new YoloWrapper("yolov5x.cfg", "x_best.weights", 0);
            }
            catch (Exception)
            {

                throw;
            }

            byte[] picBytes = GetPictureData("5_YXK54X.jpg");
            bbox_t[] result_list = yolodetector.Detect(picBytes);
                        foreach (var result in result_list)
            {
                Console.WriteLine($" hair deteteted -> x:{result.x}, y:{result.y}");
            }

            bbox_t[] result_list_2 = yolodetector.Detect("5_YXK54X.jpg");
            foreach (var result in result_list_2)
            {
                Console.WriteLine($" hair deteteted -> x:{result.x}, y:{result.y}");
            }
            Console.WriteLine("Hello World!");
        }
    }
}
