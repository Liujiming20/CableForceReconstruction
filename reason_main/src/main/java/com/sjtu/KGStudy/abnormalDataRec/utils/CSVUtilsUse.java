package com.sjtu.KGStudy.abnormalDataRec.utils;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class CSVUtilsUse {
    // HashMap用于键值转换
    private static final Map<String, String> KEY_MAPPING;
    static {
        KEY_MAPPING = new HashMap<>();
        KEY_MAPPING.put("Cable 16", "cableTension16");
        KEY_MAPPING.put("Cable 17", "cableTension17");
        KEY_MAPPING.put("Cable 18", "cableTension18");
        KEY_MAPPING.put("W 8", "environmentalTemperature8");
        KEY_MAPPING.put("W 9", "environmentalTemperature9");
        KEY_MAPPING.put("W 10", "environmentalTemperature10");
        KEY_MAPPING.put("W 11", "environmentalTemperature11");
        KEY_MAPPING.put("W 12", "environmentalTemperature12");
        KEY_MAPPING.put("W 13", "environmentalTemperature13");
        KEY_MAPPING.put("W 14", "environmentalTemperature14");
        KEY_MAPPING.put("W 15", "environmentalTemperature15");
        KEY_MAPPING.put("W 16", "environmentalTemperature16");
    }


    public static Map<String, float[]> readCSV(String filePath) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(filePath)); // 读取所有行
        if (lines.isEmpty()) {
            throw new IOException("CSV文件为空");
        }

        // 获取列名，从第二列开始
        String[] headers = lines.get(0).split(",");
        Map<String, List<Float>> tempData = new LinkedHashMap<>();
        for (int j = 1; j < headers.length; j++) { // 从第二列开始初始化
            tempData.put(headers[j].trim(), new ArrayList<>());
        }

        // 读取每行数据，从第二行开始
        for (int i = 1; i < lines.size(); i++) {
            String[] values = lines.get(i).split(",");
            for (int j = 1; j < values.length; j++) { // 从第二列开始读取数据
                String key = headers[j].trim();
                try {
                    float value = Float.parseFloat(values[j].trim()); // 转换为double
                    tempData.get(key).add(value);
                } catch (NumberFormatException e) {
                    throw new NumberFormatException("无法将值转换为double: " + values[j]);
                }
            }
        }

        // 将 List<Float> 转换为 float[]
        Map<String, float[]> data_use = new LinkedHashMap<>();
        tempData.forEach((key, list) -> {
            float[] array = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                array[i] = list.get(i);
            }
            data_use.put(KEY_MAPPING.get(key), array);
        });

        return data_use;
    }


    public static float[] calculateThresholds(float[] values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("数组不能为空");
        }

       /* float[] filteredArray = Arrays.stream(values)
                .filter(value -> Float.compare(value, -10000.0f) != 0)
                .mapToFloat(Float::floatValue)
                .toArray();*/
        // 使用 ArrayList 临时存储非异常值
        List<Float> filteredList = new ArrayList<>();
        for (float value : values) {
            if (value != -10000.0) {
                filteredList.add(value);
            }
        }
        // 将 List 转换为 float[]
        float[] results = new float[filteredList.size()];
        for (int i = 0; i < filteredList.size(); i++) {
            results[i] = filteredList.get(i);
        }


        // 计算均值
        float sum = 0.0f;
        for (float value : results) {
            sum += value;
        }
        float mean = sum / results.length;

        // 计算标准差
        float sumOfSquares = 0.0f;
        for (float value : results) {
            float diff = value - mean;
            sumOfSquares += diff * diff;
        }
        float standardDeviation = (float) Math.sqrt(sumOfSquares / results.length);

        // 返回结果
        return new float[]{mean-3*standardDeviation, mean+3*standardDeviation};
    }

}
