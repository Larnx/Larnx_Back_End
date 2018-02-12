<<<<<<< HEAD
﻿//#define Sensor1
//#define Sensor2
#define Stereo

using System;
using System.IO;
using System.Collections.Generic;
using System.Windows.Forms;
using Awaiba.Drivers.Grabbers;
using Awaiba.FrameProcessing;
using Awaiba.Media;
using Awaiba.Media.AwVideo;
using Awaiba.Media.AVI;
using Awaiba.Drivers.Grabbers.NanEye2D.FobUsb3;
using Awaiba.Algorithms;
using Awaiba.Drivers.Grabbers.Events;
using MaterialSkin;
using MaterialSkin.Controls;

namespace NanEye2D_FiberDemoUsb3_csharp
{
    public partial class Form1 : MaterialForm
    {
        /// <summary>
        /// Object that handle all the connection with the sensor
        /// </summary>
#if (Sensor1 || Sensor2)
        NanEyeFobProvider provider = new NanEyeFobProvider();
#endif
#if (Stereo)
        NanEyeFobStereoProvider provider = new NanEyeFobStereoProvider();
#endif

        /// <summary>
        ///Creates the list of AEC instances that will handle the automatic exposure control algorithm
        /// </summary>
        List<AutomaticExposureControlHardware> aec = new List<AutomaticExposureControlHardware>();

        //Video rawVideo1;
        //Video rawVideo2;
        Video processedVideo1;
        Video processedVideo2;

        public static string[] LarnxControl= Environment.GetCommandLineArgs();
        public static string savePath = LarnxControl[1];
        public static string saveName = LarnxControl[2];
        public static string ext = ".avi";
        string fullPath = Path.Combine(savePath, saveName + ext);

        public Form1()
        {
            InitializeComponent();

            // Create a material theme manager and add the form to manage (this)
            MaterialSkinManager materialSkinManager = MaterialSkinManager.Instance;
            materialSkinManager.AddFormToManage(this);
            materialSkinManager.Theme = MaterialSkinManager.Themes.LIGHT;

            // Configure color schema
            materialSkinManager.ColorScheme = new ColorScheme(
                Primary.Blue400, Primary.Blue500,
                Primary.Blue500, Accent.LightBlue200,
                TextShade.WHITE
            );

            /*** To Choose the correct files to program firmware and FPGA ***/
            ///To program the FPGA with the bin file and the FW file
            ///You can choose the folder/file combination where the files are
            provider.SetFpgaFile(@"firmware\fob_fpga_v03.bin");
            provider.SetFWFile(@"firmware\fx3_fw_2EP.img");

            /*** To choose the sensors to get data from ***/
            ///If you want to receive data from it, please put it at true, else, put it at false
            ///The sensors are organized in the code as in the Documentation, regarding the Sensor 1 and Sensor 2
            ///

#if Sensor1
            provider.Sensors = new List<bool>
            {
                true,
                false
            };
#endif
#if Sensor2
            provider.Sensors = new List<bool>
            {
                false,
                true
            };
#endif
#if Stereo
             provider.Sensors = new List<bool>
            {
                true,
                true
            };
#endif

            /*** For the Automatic Exposure Control ***/
            ///Need to create two instances, one for each connector
            ///Use the one that your sensor is plugged
            ///For more information please check the documentation on Awaiba's website
            for (int i = 0; i < 2; i++)
            {
                aec.Add(new AutomaticExposureControlHardware());
                aec[i].SensorId = i;
            }
            provider.SetAutomaticExpControl(aec);

            /*** Event Handlers to get the image and handle the exceptions from the bottom layers ***/
            provider.ImageProcessed += provider_ImageProcessed;
            provider.Exception += provider_Exception;

            /***Using Data Binding to have consistency between GUI and objects ***/
            #region Data Bindings

            #region Sensor Registers Binding
            ///Syncronization between what is already in the object and the interface
            ///To write to the sensor need to explicitly call "provider.writeregister"
            trackBar1.DataBindings.Add(new Binding("Value", provider.SensorsList[0], nameof(ISensor.Gain)));
            trackBar2.DataBindings.Add(new Binding("Value", provider.SensorsList[1], nameof(ISensor.Gain)));
            #endregion

            #region Automatic Exposure Control Algorithm
            checkBox5.DataBindings.Add(new Binding("Checked", provider.AutomaticExpControl(0), nameof(AutomaticExposureControlHardware.IsEnabled))
            {
                DataSourceUpdateMode = DataSourceUpdateMode.OnPropertyChanged
            });

            checkBox9.DataBindings.Add(new Binding("Checked", provider.AutomaticExpControl(1), nameof(AutomaticExposureControlHardware.IsEnabled))
            {
                DataSourceUpdateMode = DataSourceUpdateMode.OnPropertyChanged
            });
            #endregion

            #region Apply Color Reconstruction Algorithm
            checkBox1.DataBindings.Add(new Binding("Checked", ProcessingWrapper.pr[0].colorReconstruction, nameof(ColorReconstruction.Apply))
            {
                DataSourceUpdateMode = DataSourceUpdateMode.OnPropertyChanged
            });

            checkBox10.DataBindings.Add(new Binding("Checked", ProcessingWrapper.pr[1].colorReconstruction, nameof(ColorReconstruction.Apply))
            {
                DataSourceUpdateMode = DataSourceUpdateMode.OnPropertyChanged
            });
            #endregion

            #endregion
        }

        #region API
        private void StartEndoscope_Click(object sender, EventArgs e)
        {
            provider.StartCapture();
        }
        private void StopEndoscope_Click(object sender, EventArgs e)
        {
            provider.StopCapture();
        }

        private void provider_ImageProcessed(object sender, OnImageReceivedBitmapEventArgs e)
        {
            //Handle the image data
            if (e.SensorID == 0)
                pictureBox1.Image = ProcessingWrapper.pr[0].CreateProcessedBitmap(e.GetImageData);
            else if (e.SensorID == 1)
                pictureBox2.Image = ProcessingWrapper.pr[1].CreateProcessedBitmap(e.GetImageData);
        }
        /// <summary>
        /// This event is triggered when there is some error in the bottom layers
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void provider_Exception(object sender, OnExceptionEventArgs e)
        {
            Console.WriteLine(e.ex.Message);
        }
        #endregion

        #region Changing registers Example
        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            try
            {
                ///Can only change manually the registers if the AEC algorithm is not running
                if (provider.AutomaticExpControl(0).IsEnabled)
                {
                    Console.WriteLine("Can only change manually the registers if the AEC algorithm is not running");
                    return;
                }
                provider.WriteRegister(new RegisterPayload(0x01, true, 0, trackBar1.Value));
            }
            catch { }
        }

        private void trackBar2_Scroll(object sender, EventArgs e)
        {
            try
            {
                if (provider.AutomaticExpControl(1).IsEnabled)
                {
                    MessageBox.Show("Can only change manually the registers if the AEC algorithm is not running");
                    return;
                }
                provider.WriteRegister(new RegisterPayload(0x01, true, 1, trackBar2.Value));
            }
            catch { }
        }
        #endregion


        #region Master Recording
        private void materialRaisedButton1_Click(object sender, EventArgs e)
        {
            if (IntPtr.Size == 8)
            {
                MessageBox.Show("Saving processed avi videos not available in 64 bit mode. Please run it as x86");
                return;
            }

            if ((processedVideo1 != null) && (processedVideo2 != null))
            {
                ///Finalize the video recording
                processedVideo1.EndRecording();
                processedVideo2.EndRecording();
                masterRecord.Text = "Start Recording";
                processedVideo1.Dispose();
                processedVideo2.Dispose();
                processedVideo1 = null;
                processedVideo2 = null;
                recordingStatus.Visible = false;
                return;  
            }

            int sensorId1 = 0;
            int sensorId2 = 1;

            //if (!Int32.TryParse(comboBox1.Text, out sensorId))
            //    return;

            ///Create the raw video instance
            ///Filename
            ///Provider camera
            ///Width
            ///Height
            ///Sensor Id -> 0(Sensor1), 1(Sensor2)
            ///[Optional] Choose the frame rate. If 0 value is sent, then the video will calculate automatically when starting to record


            processedVideo1 = new AviVideoNanEyeUSB3(fullPath, provider, provider.Width, provider.Height, sensorId1, 20.0d);
            processedVideo2 = new AviVideoNanEyeUSB3(fullPath, provider, provider.Width, provider.Height, sensorId2, 20.0d);
            masterRecord.Text = "Stop Recording";
            recordingStatus.Visible = true;
        }


        #endregion

    }
}
=======
﻿//#define Sensor1
//#define Sensor2
#define Stereo

using System;
using System.IO;
using System.Collections.Generic;
using System.Windows.Forms;
using Awaiba.Drivers.Grabbers;
using Awaiba.FrameProcessing;
using Awaiba.Media;
using Awaiba.Media.AwVideo;
using Awaiba.Media.AVI;
using Awaiba.Drivers.Grabbers.NanEye2D.FobUsb3;
using Awaiba.Algorithms;
using Awaiba.Drivers.Grabbers.Events;
using MaterialSkin;
using MaterialSkin.Controls;

namespace NanEye2D_FiberDemoUsb3_csharp
{
    public partial class Form1 : MaterialForm
    {
        /// <summary>
        /// Object that handle all the connection with the sensor
        /// </summary>
#if (Sensor1 || Sensor2)
        NanEyeFobProvider provider = new NanEyeFobProvider();
#endif
#if (Stereo)
        NanEyeFobStereoProvider provider = new NanEyeFobStereoProvider();
#endif

        /// <summary>
        ///Creates the list of AEC instances that will handle the automatic exposure control algorithm
        /// </summary>
        List<AutomaticExposureControlHardware> aec = new List<AutomaticExposureControlHardware>();

        //Video rawVideo1;
        //Video rawVideo2;
        Video processedVideo1;
        Video processedVideo2;

        public static string[] LarnxControl= Environment.GetCommandLineArgs();
        public static string savePath = LarnxControl[1];
        public static string saveName = LarnxControl[2];
        public static string ext = ".avi";
        string fullPath = Path.Combine(savePath, saveName + ext);

        public Form1()
        {
            InitializeComponent();

            // Create a material theme manager and add the form to manage (this)
            MaterialSkinManager materialSkinManager = MaterialSkinManager.Instance;
            materialSkinManager.AddFormToManage(this);
            materialSkinManager.Theme = MaterialSkinManager.Themes.LIGHT;

            // Configure color schema
            materialSkinManager.ColorScheme = new ColorScheme(
                Primary.Blue400, Primary.Blue500,
                Primary.Blue500, Accent.LightBlue200,
                TextShade.WHITE
            );

            /*** To Choose the correct files to program firmware and FPGA ***/
            ///To program the FPGA with the bin file and the FW file
            ///You can choose the folder/file combination where the files are
            provider.SetFpgaFile(@"firmware\fob_fpga_v03.bin");
            provider.SetFWFile(@"firmware\fx3_fw_2EP.img");

            /*** To choose the sensors to get data from ***/
            ///If you want to receive data from it, please put it at true, else, put it at false
            ///The sensors are organized in the code as in the Documentation, regarding the Sensor 1 and Sensor 2
            ///

#if Sensor1
            provider.Sensors = new List<bool>
            {
                true,
                false
            };
#endif
#if Sensor2
            provider.Sensors = new List<bool>
            {
                false,
                true
            };
#endif
#if Stereo
             provider.Sensors = new List<bool>
            {
                true,
                true
            };
#endif

            /*** For the Automatic Exposure Control ***/
            ///Need to create two instances, one for each connector
            ///Use the one that your sensor is plugged
            ///For more information please check the documentation on Awaiba's website
            for (int i = 0; i < 2; i++)
            {
                aec.Add(new AutomaticExposureControlHardware());
                aec[i].SensorId = i;
            }
            provider.SetAutomaticExpControl(aec);

            /*** Event Handlers to get the image and handle the exceptions from the bottom layers ***/
            provider.ImageProcessed += provider_ImageProcessed;
            provider.Exception += provider_Exception;

            /***Using Data Binding to have consistency between GUI and objects ***/
            #region Data Bindings

            #region Sensor Registers Binding
            ///Syncronization between what is already in the object and the interface
            ///To write to the sensor need to explicitly call "provider.writeregister"
            trackBar1.DataBindings.Add(new Binding("Value", provider.SensorsList[0], nameof(ISensor.Gain)));
            trackBar2.DataBindings.Add(new Binding("Value", provider.SensorsList[1], nameof(ISensor.Gain)));
            #endregion

            #region Automatic Exposure Control Algorithm
            checkBox5.DataBindings.Add(new Binding("Checked", provider.AutomaticExpControl(0), nameof(AutomaticExposureControlHardware.IsEnabled))
            {
                DataSourceUpdateMode = DataSourceUpdateMode.OnPropertyChanged
            });

            checkBox9.DataBindings.Add(new Binding("Checked", provider.AutomaticExpControl(1), nameof(AutomaticExposureControlHardware.IsEnabled))
            {
                DataSourceUpdateMode = DataSourceUpdateMode.OnPropertyChanged
            });
            #endregion

            #region Apply Color Reconstruction Algorithm
            checkBox1.DataBindings.Add(new Binding("Checked", ProcessingWrapper.pr[0].colorReconstruction, nameof(ColorReconstruction.Apply))
            {
                DataSourceUpdateMode = DataSourceUpdateMode.OnPropertyChanged
            });

            checkBox10.DataBindings.Add(new Binding("Checked", ProcessingWrapper.pr[1].colorReconstruction, nameof(ColorReconstruction.Apply))
            {
                DataSourceUpdateMode = DataSourceUpdateMode.OnPropertyChanged
            });
            #endregion

            #endregion
        }

        #region API
        private void StartEndoscope_Click(object sender, EventArgs e)
        {
            provider.StartCapture();
        }
        private void StopEndoscope_Click(object sender, EventArgs e)
        {
            provider.StopCapture();
        }

        private void provider_ImageProcessed(object sender, OnImageReceivedBitmapEventArgs e)
        {
            //Handle the image data
            if (e.SensorID == 0)
                pictureBox1.Image = ProcessingWrapper.pr[0].CreateProcessedBitmap(e.GetImageData);
            else if (e.SensorID == 1)
                pictureBox2.Image = ProcessingWrapper.pr[1].CreateProcessedBitmap(e.GetImageData);
        }
        /// <summary>
        /// This event is triggered when there is some error in the bottom layers
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void provider_Exception(object sender, OnExceptionEventArgs e)
        {
            Console.WriteLine(e.ex.Message);
        }
        #endregion

        #region Changing registers Example
        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            try
            {
                ///Can only change manually the registers if the AEC algorithm is not running
                if (provider.AutomaticExpControl(0).IsEnabled)
                {
                    Console.WriteLine("Can only change manually the registers if the AEC algorithm is not running");
                    return;
                }
                provider.WriteRegister(new RegisterPayload(0x01, true, 0, trackBar1.Value));
            }
            catch { }
        }

        private void trackBar2_Scroll(object sender, EventArgs e)
        {
            try
            {
                if (provider.AutomaticExpControl(1).IsEnabled)
                {
                    MessageBox.Show("Can only change manually the registers if the AEC algorithm is not running");
                    return;
                }
                provider.WriteRegister(new RegisterPayload(0x01, true, 1, trackBar2.Value));
            }
            catch { }
        }
        #endregion


        #region Master Recording
        private void materialRaisedButton1_Click(object sender, EventArgs e)
        {
            if (IntPtr.Size == 8)
            {
                MessageBox.Show("Saving processed avi videos not available in 64 bit mode. Please run it as x86");
                return;
            }

            if ((processedVideo1 != null) && (processedVideo2 != null))
            {
                ///Finalize the video recording
                processedVideo1.EndRecording();
                processedVideo2.EndRecording();
                masterRecord.Text = "Start Recording";
                processedVideo1.Dispose();
                processedVideo2.Dispose();
                processedVideo1 = null;
                processedVideo2 = null;
                recordingStatus.Visible = false;
                return;  
            }

            int sensorId1 = 0;
            int sensorId2 = 1;

            //if (!Int32.TryParse(comboBox1.Text, out sensorId))
            //    return;

            ///Create the raw video instance
            ///Filename
            ///Provider camera
            ///Width
            ///Height
            ///Sensor Id -> 0(Sensor1), 1(Sensor2)
            ///[Optional] Choose the frame rate. If 0 value is sent, then the video will calculate automatically when starting to record


            processedVideo1 = new AviVideoNanEyeUSB3(fullPath, provider, provider.Width, provider.Height, sensorId1, 20.0d);
            processedVideo2 = new AviVideoNanEyeUSB3(fullPath, provider, provider.Width, provider.Height, sensorId2, 20.0d);
            masterRecord.Text = "Stop Recording";
            recordingStatus.Visible = true;
        }


        #endregion

    }
}
>>>>>>> master
