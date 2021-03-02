# import os
# def walkFile(file):
#     for path ,dirs,files in os.walk(file):
#         for f in files:
#             print(os.path.join(path,f))
#         # for d in dirs:
#         #     print(os.path.join(path,d))
# def main():
#     walkFile('/media/cj1/data/Kinetics-400-source/compress/train_256')
# if __name__ == '__main__':
#     main()

# import os
# path = '/media/cj1/data/Kinetics-400-source/compress/train_256/'
# for root,dirs,files in os.walk(path):
#     for dir in dirs:
#         mulu = root + '/' + dir
#         os.chdir(mulu)
#         files = os.listdir(mulu)
#         for filename in files:
#             portion = os.path.splitext(filename)
#             portion1 = os.path.splitext(portion[0])
#             if portion1[1] == '.mp4':
#                 newname = portion1[0] + '.mp4'
#                 os.rename(filename,newname)
#                 print(mulu+filename+'modify succsee!')
#             if portion[1] == '.mkv':
#                 newname = portion[0] + '.mp4'
#                 os.rename(filename,newname)
#                 print(mulu+filename+' modify succsee!')
#             if portion[1] == '.webm':
#                 newname = portion[0] + '.mp4'
#                 os.rename(filename, newname)
#                 print(mulu + filename + ' modify succsee!')
#             if portion[1] == '.mp4.webm':
#                 newname = portion[0] + '.mp4'
#                 os.rename(filename, newname)
#                 print(mulu + filename + ' modify succsee!')
#             if portion[1] == '.mp4.mp4':
#                 newname = portion[0] + '.mp4'
#                 os.rename(filename, newname)
#                 print(mulu + filename + ' modify succsee!')

# import os
# path = '/media/cj1/data/Kinetics-400-source/compress/val_256/playing_bagpipes'
# for root,dirs,files in os.walk(path):
#     for filename in files:
#         portion = os.path.splitext(filename)
#         portion1 = os.path.splitext(portion[0])
#         if portion[1] == '.mp4.mp4':
#             newname = portion[0] + '.mp4'
#             os.rename(filename, newname)
#             print( ' modify succsee!')
#         if portion1[1] == '.mp4':
#             newname = portion[0] + '.mp4'
#             os.rename(filename,newname)
#             print('modify succsee!')

# import os
# import sys
# import xlwt
# from moviepy.editor import VideoFileClip
# file_dir = u'/media/cj1/data/Kinetics-400-source/compress/train_256'
# class FileCheck():
#     def __init__(self):
#         self.file_dir = file_dir
#     #######get the video size  ################
#     def get_filesize(self,filename):
#         file_byte = os.path.getsize(filename)
#         return self.sizeConvert(file_byte)
#     ###########Unit conversion#################
#     def sizeConvert(self,size):
#         K,M,G = 1024,1024*2,1024*3
#         if size >= G:
#             return str(size/G) + 'G Bytes'
#         elif size >= M:
#             return str(size / M) + 'M Bytes'
#         elif size >= K:
#             return str(size / K) + 'K Bytes'
#         else:
#             return str(size) + ' Bytes'
#     ######get all the files under the video ###############
#     def get_all_file(self):
#         filenames = []
#         for root, dirs, files in os.walk(file_dir):
#             for dir in dirs:
#                 path = os.path.join(file_dir,dir)
#                 for paths,newdir,newfiles in os.walk(path):
#                     for file in newfiles:
#                         str = paths +"/"+ file
#                         filenames.append(str)
#             return filenames
# if __name__ == '__main__':
#     # files = FileCheck.get_all_file(file_dir)
#     # #print(files)
#     # for i in range(len(files)):
#     #     FileCheck.get_filesize(files[i])
#     # #FileCheck.get_filesize(files)
#     files = FileCheck.get_all_file(file_dir)
#
#     for i in files:
#         K = 1024
#         M = 1024**2
#         G = 1024**3
#         file_byte = os.path.getsize(i)
#         # if file_byte >= G:
#         #     print(str(file_byte / G) + 'G Bytes')
#         # elif file_byte >= M:
#         #     print(str(file_byte / M) + 'M Bytes')
#         # elif file_byte >= K:
#         #     print(str(file_byte / K) + 'K Bytes')
#         # else:
#         #     print(str(file_byte) + ' Bytes')
#         if file_byte == 0:
#             print(i)
#         #FileCheck.get_filesize(i)




# import os
# path = '/media/cj1/data/kinetics-400-fps/image/parkour'
# for root,dirs,files in os.walk(path):
#     for dir in dirs:
#         #print([i for i,v in enumerate(dir) if v == '_'])
#         #char_ind = [i for i,v in  enumerate(dir) if v== '_']
#         print(dir[:11],len(str(dir)))

import os
path = '/media/cj1/data/kinetics-400-fps/image/'
for root,dirs,files in os.walk(path):
    for dir in dirs:
        new_path = os.path.join(root,dir)
        for new_root,new_dirs,new_files in os.walk(new_path):
            for new_dir in new_dirs:
                mulu = new_root
                os.chdir(mulu )
                dirname = new_dir
                print(dirname)
                new_dirname = dirname[:11]
                print(new_dirname)
                os.rename(dirname, new_dirname)
                print(mulu +  dirname + 'is ok!')

        # os.rename(dirname,new_dirname)
        # print(mulu +'is ok!')


