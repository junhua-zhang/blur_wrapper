using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace ResNet
{
    public class ResNet18Wrapper : IDisposable
    {
        private const string ResNetLibraryName = "resnet18.dll";
        private const int MaxObjects = 1000;

        [DllImport(ResNetLibraryName, EntryPoint = "init")]
        private static extern int InitializeResNet18(string configurationFilename, string weightsFilename, int gpu);

        [DllImport(ResNetLibraryName, EntryPoint = "class_image")]
        private static extern void ClassImage(string filename, ref float conf);

        [DllImport(ResNetLibraryName, EntryPoint = "class_mat")]
        private static extern void ClassImage(IntPtr pArray, int nSize, ref float conf);

        [DllImport(ResNetLibraryName, EntryPoint = "dispose")]
        private static extern int DisposeResNet18();

        public ResNet18Wrapper(string configurationFilename, string weightsFilename, int gpu)
        {
            InitializeResNet18(configurationFilename, weightsFilename, gpu);
        }

        public void Dispose()
        {
            DisposeResNet18();
        }

        public float ClassBlur(string filename)
        {
            float conf = 0;
            ClassImage(filename, ref conf);

            return conf;
        }


        public float ClassBlur(byte[] imageData)
        {

            var size = Marshal.SizeOf(imageData[0]) * imageData.Length;
            var pnt = Marshal.AllocHGlobal(size);
            int blurry;
            float conf = 0;

            try
            {
                // Copy the array to unmanaged memory.
                Marshal.Copy(imageData, 0, pnt, imageData.Length);
                ClassImage(pnt, imageData.Length, ref conf);
            }
            catch (Exception exception)
            {
                return 0;
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

            return conf;
        }

    }
}
