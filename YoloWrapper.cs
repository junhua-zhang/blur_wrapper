using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Darknet
{
    [StructLayout(LayoutKind.Sequential)]
    public struct bbox_t
    {
        public UInt32 x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
        public float prob;                 // confidence - probability that the object was found correctly
        public UInt32 obj_id;        // class of object - from range [0, classes-1]
        public UInt32 track_id;
        public UInt32 frames_counter;
        public float x_3d, y_3d, z_3d;  // 3-D coordinates, if there is used 3D-stereo camera
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct BboxContainer
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1000)]
        public bbox_t[] candidates;
    }

    public class YoloWrapper : IDisposable
    {
        private const string YoloLibraryName = "yolov5.dll";
        //private const int MaxObjects = 1000;

        [DllImport(YoloLibraryName, EntryPoint = "init")]
        private static extern int InitializeYolo(string configurationFilename, string weightsFilename, int gpu, int classNum);

        [DllImport(YoloLibraryName, EntryPoint = "detect_image")]
        private static extern int DetectImage(string filename, ref BboxContainer container);

        [DllImport(YoloLibraryName, EntryPoint = "detect_mat")]
        private static extern int DetectImage(IntPtr pArray, int nSize, ref BboxContainer container);

        [DllImport(YoloLibraryName, EntryPoint = "dispose")]
        private static extern int DisposeYolo();


        public YoloWrapper(string configurationFilename, string weightsFilename, int gpu, int classNum)
        {
            InitializeYolo(configurationFilename, weightsFilename, gpu, classNum);
        }

        public void Dispose()
        {
            DisposeYolo();
        }

        public bbox_t[] Detect(string filename)
        {
            var container = new BboxContainer();
            var count = DetectImage(filename, ref container);

            return container.candidates;
        }


        public bbox_t[] Detect(byte[] imageData)
        {
            var container = new BboxContainer();

            var size = Marshal.SizeOf(imageData[0]) * imageData.Length;
            var pnt = Marshal.AllocHGlobal(size);

            try
            {
                // Copy the array to unmanaged memory.
                Marshal.Copy(imageData, 0, pnt, imageData.Length);
                var count = DetectImage(pnt, imageData.Length, ref container);
                if (count == -1)
                {
                    throw new NotSupportedException($"{YoloLibraryName} has no OpenCV support");
                }
            }
            catch (Exception exception)
            {
                return null;
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

            return container.candidates;
        }

        public async Task<bbox_t[]> DetectAsync(byte[] imageData)
        {
            return await Task.Run(() =>
            {
                var container = new BboxContainer();
                var size = Marshal.SizeOf(imageData[0]) * imageData.Length;
                var pnt = Marshal.AllocHGlobal(size);

                try
                {
                    // Copy the array to unmanaged
                    Marshal.Copy(imageData, 0, pnt, imageData.Length);

                    var count = DetectImage(pnt, imageData.Length, ref container);

                    if (count == -1)
                    {
                        throw new NotSupportedException($"{YoloLibraryName} has no OpenCv support");
                    }
                }
                catch ( Exception exception)
                {
                    return null;
                }
                finally
                {
                    Marshal.FreeHGlobal(pnt);
                }
                return container.candidates;
            });
        }
    }

    public class IdWrapper : IDisposable
    {
        private const string IdLibraryName = "id_yolov5.dll";
        //private const int MaxObjects = 1000;

        [DllImport(IdLibraryName, EntryPoint = "init")]
        private static extern int InitializeId(string configurationFilename, string weightsFilename, int gpu, int classNum);

        [DllImport(IdLibraryName, EntryPoint = "detect_image")]
        private static extern int DetectImage(string filename, ref BboxContainer container);

        [DllImport(IdLibraryName, EntryPoint = "detect_mat")]
        private static extern int DetectImage(IntPtr pArray, int nSize, ref BboxContainer container);

        [DllImport(IdLibraryName, EntryPoint = "dispose")]
        private static extern int DisposeId();

        //[StructLayout(LayoutKind.Sequential)]
        //public struct bbox_t
        //{
        //    public UInt32 x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
        //    public float prob;                 // confidence - probability that the object was found correctly
        //    public UInt32 obj_id;        // class of object - from range [0, classes-1]
        //    public UInt32 track_id;
        //    public UInt32 frames_counter;
        //};

        //[StructLayout(LayoutKind.Sequential)]
        //public struct BboxContainer
        //{
        //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxObjects)]
        //    public bbox_t[] candidates;
        //}

        public IdWrapper(string configurationFilename, string weightsFilename, int gpu, int classNum)
        {
            InitializeId(configurationFilename, weightsFilename, gpu, classNum);
        }

        public void Dispose()
        {
            DisposeId();
        }

        public bbox_t[] Detect(string filename)
        {
            var container = new BboxContainer();
            var count = DetectImage(filename, ref container);

            return container.candidates;
        }


        public bbox_t[] Detect(byte[] imageData)
        {
            var container = new BboxContainer();

            var size = Marshal.SizeOf(imageData[0]) * imageData.Length;
            var pnt = Marshal.AllocHGlobal(size);

            try
            {
                // Copy the array to unmanaged memory.
                Marshal.Copy(imageData, 0, pnt, imageData.Length);
                var count = DetectImage(pnt, imageData.Length, ref container);
                if (count == -1)
                {
                    throw new NotSupportedException($"{IdLibraryName} has no OpenCV support");
                }
            }
            catch (Exception exception)
            {
                return null;
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

            return container.candidates;
        }

        public async Task<bbox_t[]> DetectAsync(byte[] imageData)
        {
            return await Task.Run(() =>
            {
                var container = new BboxContainer();
                var size = Marshal.SizeOf(imageData[0]) * imageData.Length;
                var pnt = Marshal.AllocHGlobal(size);

                try
                {
                    // Copy the array to unmanaged
                    Marshal.Copy(imageData, 0, pnt, imageData.Length);

                    var count = DetectImage(pnt, imageData.Length, ref container);

                    if (count == -1)
                    {
                        throw new NotSupportedException($"{IdLibraryName} has no OpenCv support");
                    }
                }
                catch ( Exception exception)
                {
                    return null;
                }
                finally
                {
                    Marshal.FreeHGlobal(pnt);
                }
                return container.candidates;
            });
        }
    }
}
