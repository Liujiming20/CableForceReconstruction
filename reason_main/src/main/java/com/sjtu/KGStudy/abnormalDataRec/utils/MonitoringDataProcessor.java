package com.sjtu.KGStudy.abnormalDataRec.utils;

import java.io.*;
import java.sql.*;
import java.util.*;
import org.json.JSONObject;

public class MonitoringDataProcessor {
    // JDBC连接信息
    private static final String JDBC_URL = "jdbc:mysql://localhost:3306/sjtucablenetdatadiagnosis";
    private static final String JDBC_USER = "root";
    private static final String JDBC_PASSWORD = "980811";

    private String accTableName;
    public MonitoringDataProcessor(String accTableName) {
        this.accTableName = accTableName;
    }

    private static final String OUTPUT_DIRECTORY = "G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData";

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

    public String processRowByIndex(int rowIndex) {
        String jonsFileName = "";

        try (Connection connection = DriverManager.getConnection(JDBC_URL, JDBC_USER, JDBC_PASSWORD)) {
            String query = "SELECT * FROM " + accTableName;
            try (
                    // PreparedStatement statement = connection.prepareStatement(query);
                    PreparedStatement statement = connection.prepareStatement(query, ResultSet.TYPE_SCROLL_INSENSITIVE, ResultSet.CONCUR_READ_ONLY);

                    ResultSet resultSet = statement.executeQuery()) {

                // 获取列名
                ResultSetMetaData metaData = resultSet.getMetaData();
                int columnCount = metaData.getColumnCount();
                List<String> columnNames = new ArrayList<>();
                for (int i = 1; i <= columnCount; i++) {
                    columnNames.add(metaData.getColumnName(i));
                }

                // 定位到目标行
                int currentRow = 1;
                while (currentRow < rowIndex && resultSet.next()) {
                    currentRow++;
                }

                if (resultSet.next()) {
                    // 读取目标行数据
                    Map<String, String> targetRow = new LinkedHashMap<>();
                    for (int i = 1; i <= columnCount; i++) {
                        targetRow.put(columnNames.get(i - 1), resultSet.getString(i));
                    }

                    // 写入JSON文件
                    jonsFileName = saveRowAsJson(targetRow, columnNames);

                    // 读取倒数50行数据
                    List<Map<String, String>> last50Rows = new ArrayList<>();

                    // resultSet.beforeFirst();
                    resultSet.relative(-51); // 回退 倒数50 行
                    String valueCol = " ";
                    for (int index = 1; index <= 50; index++) {
                        if (resultSet.next()) {
                            Map<String, String> row = new LinkedHashMap<>();
                            for (int i = 1; i <= columnCount; i++) {
                                if (resultSet.getString(i) == null){
                                    valueCol = "-10000.0";
                                }else {
                                    valueCol = resultSet.getString(i);
                                }
                                row.put(columnNames.get(i - 1), valueCol);
                            }
                            last50Rows.add(row);
                        }
                    }

                    // 写入CSV文件
                    saveRowsAsCsv(last50Rows, columnNames);
                } else {
                    System.out.println("行号超出范围或数据不存在。");
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }

        return jonsFileName;
    }

    // 将行数据保存为JSON文件
    private String saveRowAsJson(Map<String, String> row, List<String> columnNames) {
        String timestamp = row.get(columnNames.get(0)).replaceAll("[^a-zA-Z0-9]", "-"); // 替换特殊字符
        JSONObject jsonObject = new JSONObject();

        double value = 0.0;
        for (Map.Entry<String, String> entry : row.entrySet()) {
            String columnName = entry.getKey();
            if (!columnName.equals(columnNames.get(0))) { // 跳过时间戳列
                String mappedKey = KEY_MAPPING.getOrDefault(columnName, columnName); // 使用映射后的键
                try {
                    if (entry.getValue() == null){
                        value = -10000.0;
                    }else {
                        value = Double.parseDouble(entry.getValue()); // 转换为 double
                    }

                    jsonObject.put(mappedKey,  Arrays.asList(row.get(columnNames.get(0)), value));
                } catch (NumberFormatException e) {
                    System.out.println("无法将值转换为 double，列名：" + columnName + "，值：" + entry.getValue());
                    // 可选择忽略或赋默认值，例如 jsonObject.put(mappedKey, 0.0);
                }
            }
        }

        File directory = new File(OUTPUT_DIRECTORY);
        if (!directory.exists()) {
            directory.mkdirs();
        }

        try (FileWriter file = new FileWriter(new File(directory, timestamp + ".json"))) {
            file.write(jsonObject.toString(4)); // 格式化JSON
        } catch (IOException e) {
            e.printStackTrace();
        }

        return timestamp;
    }

    // 将多行数据保存为CSV文件
    private void saveRowsAsCsv(List<Map<String, String>> rows, List<String> columnNames) {
        File directory = new File(OUTPUT_DIRECTORY);
        if (!directory.exists()) {
            directory.mkdirs();
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(directory, "last_50_rows.csv")))) {
            // 写入列名
            writer.write(String.join(",", columnNames));
            writer.newLine();

            // 写入行数据
            for (Map<String, String> row : rows) {
                List<String> values = new ArrayList<>();
                for (String column : columnNames) {
                    values.add(row.getOrDefault(column, ""));
                }
                writer.write(String.join(",", values));
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
