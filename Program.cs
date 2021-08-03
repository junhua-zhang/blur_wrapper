using System;
using System.IO;
using Darknet;
using static Darknet.YoloWrapper;
using ResNet;
using static ResNet.ResNet18Wrapper;

using OpenCvSharp;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Drawing;

namespace YoloTest_CS
{
    class YoloResult
    {
        public int x { get; set; }
        public int y { get; set; }
        public int w { get; set; }
        public int h { get; set; }
        public float prob { get; set; }

    }
    class Program
    {
        static YoloWrapper yolodetector;
        static ResNet18Wrapper resnetclassificator;
        const float conf_thres = 0.25f;

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
            string path = args[0];
            string output_path = args[1];
            DirectoryInfo root = new DirectoryInfo(path);
            FileInfo[] file_info_list = root.GetFiles();

            //load roi detection model yolov5s, and blur classification model resnet18
            DateTime begin_load_model = DateTime.Now;
            try
            {
                yolodetector = new YoloWrapper("yolov5s", "../yolov5s.wts", 0);
                resnetclassificator = new ResNet18Wrapper("resnet18", "../resnet18.wts", 0);
            }
            catch (Exception)
            {
                throw;
            }
            DateTime end_load_model = DateTime.Now;
            Console.WriteLine("+++++++++++++++++++++++++++++++++++");
            Console.WriteLine($"Time consumed for loading model: {(end_load_model - begin_load_model).TotalMilliseconds} ms");
            Console.WriteLine("+++++++++++++++++++++++++++++++++++");

            DateTime begin_detect_class = DateTime.Now;
            if (!Directory.Exists(output_path))
            {
                 Directory.CreateDirectory(output_path);
            }
            foreach( var file_info in file_info_list)
            {
                DetectROIClassBlur(path, file_info.Name, output_path);
                Console.WriteLine("****************************************");
                Console.WriteLine();
            }
            DateTime end_detect_class = DateTime.Now;
            Console.WriteLine("+++++++++++++++++++++++++++++++++++");
            Console.WriteLine($"Time consumed for evaluate {file_info_list.Length} images: {(end_detect_class - begin_detect_class).TotalMilliseconds} ms");
            Console.WriteLine("+++++++++++++++++++++++++++++++++++");


            //dispose of the two models
            yolodetector.Dispose();
            resnetclassificator.Dispose();
        } 


        /*
        static void Main(string[] args)
        {

            string path = $"{Environment.CurrentDirectory}\\test";
            DirectoryInfo root = new DirectoryInfo(path);
            FileInfo[] dics = root.GetFiles();

            try
            {
                yolodetector = new YoloWrapper("yolov5s", "../yolov5s.wts", 0);
                resnetclassificator = new ResNet18Wrapper("resnet18", "../resnet18.wts", 0);
            }
            catch (Exception)
            {
                throw;
            }

            int i = 0;
            int task_count = 2;
            List<Task> task_list = new List<Task>();

            for (i = 0; i < task_count; i++)
            {
                foreach (var ca in dics)
                {
                    task_list.Add
                        (Task.Run(() =>
                        {
                            Test(ca.FullName, "task 1");
                        }));
                    task_list.Add
                        (Task.Run(() =>
                        {
                            Test(ca.FullName, "task 2");
                        }));
                }

            }
            Task.WaitAll(task_list.ToArray());
            yolodetector.Dispose();
            resnetclassificator.Dispose();
        }
        */

        public static void DetectROIClassBlur(string path, string image_name, string output_path, bool save_plot=false)
        {
            Console.WriteLine($"image file name: {path}//{image_name}");
            //read images
            Mat img = Cv2.ImRead($"{path}//{image_name}");
            byte[] picBytes = img.ToBytes();

            //detection of roi using yolov5s
            bbox_t[] result_list;
            lock(yolodetector)
            {
                result_list = yolodetector.Detect(picBytes);
            }

            List<YoloResult> yoloResults = new List<YoloResult>();
            List<(int, float)> blurryResults = new List<(int, float)>();

            //convert roi result from bbox_t[] to YoloResult
            foreach (var b in result_list)
            {
                if (b.h == 0)
                {
                    break;
                }
                if (b.prob < conf_thres)
                {
                    continue;
                }
                yoloResults.Add(new YoloResult
                {
                    x = (int)b.x,
                    y = (int)b.y,
                    w = (int)b.w,
                    h = (int)b.h,
                    prob = b.prob
                });
            }

            Mat plot_img = null;
            //if (save_plot)
            //{
            //    plot_img = img.Clone();
            //}
            //start blur classification
            Console.WriteLine("start blurry classification");
            foreach (var result in yoloResults)
            {
                Console.WriteLine($"roi deteteted -> x:{result.x}, y:{result.y}");

                //crop the two patches using roi result from yolov5s in turn
                var roi = new OpenCvSharp.Rect(result.x, result.y, result.w, result.h);
                Mat patch = new Mat(img, roi);
                byte[] patchBytes = patch.ToBytes();
                //check if the patch is blurry using resnet18
                //result is (blurry, conf), blurry=1 means blurry, blurry=0 means not blurry
                //conf is the confidence of current classification result
                int blurry;
                float conf;
                lock(resnetclassificator){
                    (blurry, conf) = resnetclassificator.ClassBlur(patchBytes);
                }
                bool isBlurry = blurry > 0 ? true : false;
                Console.WriteLine($"blurry -> {isBlurry} with confidence -> {conf}");
                blurryResults.Add((blurry, conf));
                save_plot = save_plot || isBlurry;
                //plot of roi detection
                if (save_plot)
                {
                    if(plot_img == null)
                    {
                        plot_img = img.Clone();
                    }
                    plot_img.Rectangle(new OpenCvSharp.Rect(result.x, result.y, result.w, result.h), Scalar.Red, 3);
                    plot_img.PutText(isBlurry.ToString() + " " + conf.ToString("0.00"), new OpenCvSharp.Point(result.x, result.y+100), HersheyFonts.Italic, 4, Scalar.Red, 4);
                }
            }
            if (save_plot)
            {
                Cv2.ImWrite($"{output_path}//_{image_name}", plot_img);
            }

            //Cv2.NamedWindow("img", 0);
            //Cv2.ResizeWindow("img", 2592/4, 1944/4);
            //Cv2.ImShow("img", plot_img);
            //Cv2.WaitKey();
        }
    }

}
