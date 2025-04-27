package com.sjtu.KGStudy.abnormalDataRec.utils;

import java.io.File;

public class FileCleanUtils {
    public static void deleteFilesInFolder(File folder) {
        File[] files = folder.listFiles();
        if (files != null) {
            for (File file : files) {
                // 判断是否为文件
                if (file.isFile()) {
                    file.delete();
                }
            }
        }
    }
}
