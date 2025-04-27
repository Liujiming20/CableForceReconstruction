package com.sjtu.KGStudy.abnormalDataRec.utils;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.sql.Timestamp;

import java.io.*;
import java.sql.*;
import java.util.*;

import org.json.JSONObject;

public class UpdateEnvTempCurrentAbnormalObs {
    // JDBC连接信息
    private static final String JDBC_URL = "jdbc:mysql://localhost:3306/sjtucablenetdatadiagnosis";
    private static final String JDBC_USER = "root";
    private static final String JDBC_PASSWORD = "980811";

    private String accTableName;
    public UpdateEnvTempCurrentAbnormalObs(String accTableName) {
        this.accTableName = accTableName;
    }

    // HashMap用于键值转换
    private static final Map<String, String> KEY_MAPPING;
    static {
        KEY_MAPPING = new HashMap<>();
        KEY_MAPPING.put( "environmentalTemperature8", "W 8");
        KEY_MAPPING.put( "environmentalTemperature9", "W 9");
        KEY_MAPPING.put( "environmentalTemperature10", "W 10");
        KEY_MAPPING.put( "environmentalTemperature11", "W 11");
        KEY_MAPPING.put( "environmentalTemperature12", "W 12");
        KEY_MAPPING.put( "environmentalTemperature13", "W 13");
        KEY_MAPPING.put( "environmentalTemperature14", "W 14");
        KEY_MAPPING.put( "environmentalTemperature15", "W 15");
        KEY_MAPPING.put( "environmentalTemperature16", "W 16");
    }

    public void updateCellValue(String columnNameOri, double markedValue, String timeStampStr, boolean mappingOption) {
        String columnName = null;
        if (mappingOption) {
            columnName = KEY_MAPPING.get(columnNameOri);
        }else {
            columnName = columnNameOri;
        }

        Timestamp timeCurrent = convertToTimestamp(timeStampStr);

        // SQL 更新语句
        String updateSQL = "UPDATE " + accTableName + " SET `" + columnName + "` = ? WHERE Time = ?";

        // 数据库连接对象
        try (Connection connection = DriverManager.getConnection(JDBC_URL, JDBC_USER, JDBC_PASSWORD);
             PreparedStatement preparedStatement = connection.prepareStatement(updateSQL)) {

            // 设置更新的值
            preparedStatement.setDouble(1, markedValue);  // 设置新的单元格值
            preparedStatement.setTimestamp(2, timeCurrent);  // 设置匹配的时间戳

            // 执行更新操作
            preparedStatement.executeUpdate();

            // System.out.println("数据已成功更新！");

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void updateRecoveredData(String jsonFilePath, String timeStampStr) {
        // 读取 JSON 文件内容为字符串
        String content = null;
        try {
            content = new String(Files.readAllBytes(Paths.get(jsonFilePath)));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 使用 JSONObject 解析 JSON 字符串
        JSONObject jsonObject = null;
        if (content == null) {
            return;
        }else {
            jsonObject = new JSONObject(content);
        }

        // 将 JSONObject 转为 Map
        Map<String, Object> dataJson = jsonObject.toMap();

        // 获取keys
        for (Map.Entry<String, Object> entry : dataJson.entrySet()) {
            String key = entry.getKey();
            if (key.contains("DisabledReconstructionModeShape")) {
                System.out.println("当前系统监测缺失的环境监测数据过多，软件已无法提供可靠的数据恢复服务，请及时核查传感器服役状态！");
                return;
            }else {
                double updated_value = Double.parseDouble(entry.getValue().toString());
                updateCellValue(key, updated_value, timeStampStr, false);
            }
        }
    }


    // 将日期字符串转换为 Timestamp 的私有方法
    private Timestamp convertToTimestamp(String dateString) {
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
        try {
            Date date = formatter.parse(dateString);
            return new Timestamp(date.getTime());
        } catch (ParseException e) {
            e.printStackTrace();
            return null; // 返回 null 以指示错误
        }
    }
}
