from pymelsec import Type3E
from pymelsec.constants import DT
import threading
import os
import time
import pandas as pd
import cv2
import neoapi
#------------------------------------------------------------------------------------
uri_machine_input_value = pd.read_csv(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\Document\no_rub_input_value.csv", index_col=0)
dinput_addr = [*uri_machine_input_value['ID_Input']]
dinput_type = [*uri_machine_input_value['Input_Type']]
dinput_name = [name.strip() for name in [*uri_machine_input_value['Input_Name']]]
dinput_index = [*uri_machine_input_value['Input_Index']]
dinput_lenght = len(dinput_addr)
dinput_old = [-1]*dinput_lenght

class ProductionSystem:
    def __init__(self):
        # self.plc_host = plc_host
        # self.plc_port = plc_port
        # self.plc_type = plc_type
        self.plc = None
        self.camera = None
        self.lock = threading.Lock()
        self.triggered = threading.Event()  # Sự kiện kích hoạt khi lấy biến từ PLC thành công
        self.results = []
        self.error_logs = [] 
        self.plc_connected = True #PLC luon ket noi
        self.camera_connected = False
        self.image_directory = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\GreenDatasetTask08012025"
        self.processing = False
        self.last_trigger_state = 0
        self.trigger = None  # Initialize trigger to None
        self.timestamp_S3_in_19 = None
        self.timestamp_S3_in_22 = None
    # 1. Kết nối với PLC
    
    # def PLCConnection(self):
    #     if self.plc_connected:
    #         return True
    #     # try:
    #     #     self.plc = Type3E(host=self.plc_host, port=self.plc_port, plc_type=self.plc_type)
    #     #     self.plc.set_access_opt(comm_type='binary')
    #     #     self.plc.connect(ip=self.plc_host, port=self.plc_port)
    #     #     print("Connecting to PLC.")
    #     #     self.plc_connected = True
    #     #     return True
    #     # except Exception as e:
    #     #     print(f"Error connecting to PLC: {e}")
    #     #     self.error_logs.append(str(e))
    #     #     self.plc_connected = False
    #     #     return False
    # # 1.1 Kiểm tra kết nối với PLC
    # def CheckPLCConnection(self):
    #     if self.plc_connected:
    #         try:
    #             # Đọc một biến trạng thái từ PLC để kiểm tra kết nối
    #             read_result = self.plc.batch_read(ref_device='X0',
    #                                                 read_size=50, 
    #                                                 data_type=DT.BIT)
    #             self.plc_connected = True       
    #             print("PLC connection is active.")    
    #         except Exception:
    #             print("PLC connection lost.")
    #             self.error_logs.append("PLC connection lost.")
    #             self.plc_connected = False  # mất kết nối
    #     else:
    #         # Nếu không có kết nối thì thử kết nối lại
    #         print("Attempting to reconnect to PLC...")
    #         self.PLCConnection()

    # 2. Kết nối camera
    def CameraConnection(self):
        # Kiểm tra xem camera đã kết nối hay chưa
        if self.camera_connected:
            return True
        try:
            self.camera = neoapi.Cam()
            self.camera.Connect()   
            if self.camera.IsConnected():
                print("Connecting to Camera.")
                self.camera_connected = True
                return True
            else:
                raise Exception("Camera not connected.")
        except (neoapi.NeoException, Exception) as e:
            print(f"Error checking camera connection: {e}")
            self.error_logs.append(str(e))
            self.camera_connected = False
            return False

    # 2.1 Kết nối camera
    def CheckCameraConnection(self):
        if self.camera_connected:
            try: 
                ip_address = self.camera.GetFeature("GevCurrentIPAddress").GetString()
                #print("Camera connection is active.")
                self.camera_connected = True
            except (neoapi.NeoException, Exception) as e:
                print(f"Error checking camera connection: {e}")
                self.error_logs.append(str(e))
                self.camera_connected = False
                
                # Nếu không có kết nối thì thử kết nối lại
                print("Attempting to reconnect to Camera...")
                self.CameraConnection()  # Gọi lại hàm để thử kết nối lại
                return False

    # 2.2 Thiết lập thông số camera   
    def ConfigCamera(self):
        if self.CameraConnection():  # Gọi kiểm tra kết nối trước khi thiết lập thông số
            try:
                # Thiết lập các thông số cho camera
                #self.camera.f.ExposureTime.Set(1500)
                self.camera.f.ExposureTime.Set(1000)
 
                self.camera.f.Width.Set(2592)
                self.camera.f.Height.Set(2048)
                self.camera.f.OffsetX.Set(0)
                self.camera.f.Gain.Set(0)
                self.camera.f.PixelFormat.Set(neoapi.PixelFormat_BayerRG8)
                self.camera.SetSynchronFeatureMode(False) 

                #Tắt sharpending
                self.camera.f.TriggerMode.value = neoapi.TriggerMode_On
                self.trigger = self.camera.f.TriggerSoftware
                if self.camera.HasFeature("SharpeningEnable"):
                    self.camera.SetFeature("SharpeningEnable", False)
                if self.camera.HasFeature("NoiseReductionEnable"):
                    self.camera.SetFeature("NoiseReductionEnable", False)

                #Bật chế độ use optimal buffer size
                if self.camera.HasFeature("UseOptimalBufferSize"):
                    self.camera.SetFeature("UseOptimalBufferSize", True)
                return True
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Camera connection failed. Cannot configure settings.")
            return False

    # 3. Lấy biến PLC để kích hoạt camera trigger
    def GetPlcVariableForTrigger(self):
        while True:
            #Nhập input từ terminal bằng 1
            x = input("Enter value for x: ")
            if x == '1':
                self.triggered = True
                print("Trigger set to True")
            else:
                self.triggered = False
                print("Trigger set to False")
            # for i in range(dinput_lenght):
            #     try:
            #         with self.lock:
            #             # Đọc biến từ PLC
            #             read_result = self.plc.batch_read(ref_device='X0', read_size=50, data_type=DT.BIT)
            #             RealValue1 = int(read_result[29].value)

            #             if dinput_old[20] != RealValue1:
            #                 dinput_old[20] = RealValue1
            #                 print("S3/in/19", RealValue1)
                                
            #             trigger_value = RealValue1
            #             # Break the loop or return from the function to stop the trigger
            #             if trigger_value == 1  and self.last_trigger_state == 0:  
            #                 self.triggered.set()  # Đặt cờ sự kiện kích hoạt
            #                 self.processing = True
            #             elif trigger_value == 0:
            #                 self.processing = False  # Reset lại trạng thái xử lý khi trigger_value là 0

            #             self.last_trigger_state = trigger_value  # Cập nhật trạng thái trước của trigger_value
            #             return
                    
            #     except Exception as e:
            #         #print("Failed to get PLC variable:", e)
            #         pass
   
    def GetNextFilename(self):
        # Tìm số thứ tự tệp cuối cùng trong thư mục
        existing_files = sorted(
            [f for f in os.listdir(self.image_directory) if f.startswith("IMG_") and f.endswith(".bmp")]
        )
        
        if existing_files:
            # Lấy số thứ tự của tệp cuối cùng và tăng lên 1
            last_file = existing_files[-1]
            last_number = int(last_file[4:8])  # Lấy số từ "IMG_0001.bmp"
            next_number = (last_number + 1) % 10000  # Giới hạn từ 0000 đến 9999
        else:
            next_number = 1

        return os.path.join(self.image_directory, f"IMG_{next_number:04d}.bmp")      

    # Phương thức chạy hệ thống
    def Run(self):
        # Tạo và khởi động luồng cho kết nối PLC
        # plc_thread = threading.Thread(target=self.PLCConnection)
        # plc_thread.start()

         # Tạo và khởi động luồng cho kiểm tra camera
        camera_thread = threading.Thread(target=self.CameraConnection)
        camera_thread.start()

        # Chờ cho hai luồng kết nối PLC và camera hoàn thành
        #plc_thread.join()
        camera_thread.join()

        connection_monitor_thread = threading.Thread(target=self.ConnectionMonitor)
        connection_monitor_thread.start()

        # if self.PLCConnection() and self.CameraConnection():
        if self.CameraConnection():
            print("System is fully connected.")

            # Gọi hàm kiểm tra kết nối PLC
            #self.CheckPLCConnection()

            # Thiết lập các thông số camera sau khi kiểm tra kết nối thành công
            camera_configured = self.ConfigCamera()

            if not camera_configured:
                print("Camera configuration failed.")
                return  # Thoát nếu cấu hình camera không thành công

            while True:
                self.GetPlcVariableForTrigger()
                print("Waiting for next trigger")

                if self.triggered.is_set():
                    self.processing = True
                    print("Trigger activated, processing images...")
                    with self.lock:
                        try:
                            start = time.time()
                            #time.sleep(0.3)
                            time.sleep(1.195)
                            # Execute the camera trigger
                            if self.trigger:
                                self.trigger.Execute()
                            else:
                                print("Trigger is not initialized.")
                            time.sleep(0.05) 
                            img_camera = self.camera.GetImage()
                            file_path = self.GetNextFilename()
                            img_camera.Save(file_path)
                            print(f"Image saved at {file_path}")
                            end = time.time()
                            elapsed_time = end - start 
                            print(f'Time: {elapsed_time:.5f} giây')
                            img = cv2.imread(file_path)
                            cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
                            cv2.imshow("finalImg",img)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        
                            # Tạo và khởi động hai luồng song song cho task 1 và task 2
                            # task1_thread = threading.Thread(target=self.Task1, args=(img,))
                            # task2_thread = threading.Thread(target=self.Task2, args=(img,))

                            # task1_thread.start()
                            # task2_thread.start()

                            # # Kiểm tra kết quả khi các luồng hoàn thành
                            # while task1_thread.is_alive() or task2_thread.is_alive():
                            #     time.sleep(0.1)  # Tạm dừng một chút trước khi kiểm tra lại

                            # # Ghi nhận kết quả từ các luồng
                            # self.RecordResults(task1_thread, task2_thread)

                            # Reset trigger sau khi hoàn thành xử lý
                            self.triggered.clear()  # Đặt lại trigger để chờ lần kích hoạt tiếp theo
                            self.processing = False  
                        except Exception as e:
                            print(f"Error capturing or saving image: {e}")
                        finally:
                            # Reset trigger sau khi hoàn thành xử lý
                            self.triggered.clear()
                            self.processing = False
                # else:
                #     time.sleep(0.01)
                    
        else:
            print("System setup failed.")

    # 4. Task1: Counting
    def Task1(self, image):
        self.triggered.wait()  # Chờ cho đến khi trigger được kích hoạt
        print("Starting task 1 preprocessing...")
        
        print("Task 1 preprocessing complete.")
        return
    
    # 5. Task2: Infect detection
    def Task2(self, image):
        self.triggered.wait()  # Chờ cho đến khi trigger được kích hoạt
        print("Starting task 2 preprocessing...")

        print("Task 2 preprocessing complete.")
        return
    
    def RecordResults(self, task1_thread, task2_thread):
        with self.lock:
            # Ghi nhận kết quả từ các luồng
            if task1_thread.is_alive() or task2_thread.is_alive():
                print("Some tasks are still running. Results will be recorded later.")
            else:
                if task1_thread in self.results:
                    print(f"Result from Task 1: {self.results[task1_thread]}")
                if task2_thread in self.results:
                    print(f"Result from Task 2: {self.results[task2_thread]}")

    def ConnectionMonitor(self):
        # Luồng kiểm tra kết nối định kỳ.
        while True:
            # Kiểm tra kết nối PLC
            # self.CheckPLCConnection()
            # if not self.plc_connected:
            #     print("PLC connection lost. Attempting to reconnect...")
            #     self.PLCConnection()
            self.CheckCameraConnection()
            # Kiểm tra kết nối camera
            if not self.camera_connected:
                print("Camera connection lost. Attempting to reconnect...")
                self.CameraConnection()
            
            time.sleep(5)  # Đợi 5 giây trước khi kiểm tra lại

#Main
if __name__ == '__main__':
    system = ProductionSystem()
    #production_system.check_trigger()
    system.Run()
    # production_system = ProductionSystem()
    # production_system.check_trigger()
