package com.sjtu.KGStudy.abnormalDataRec.utils;

import com.sjtu.KGStudy.utils.PrefixesDel;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Resource;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RecModeReportDel {
    public static void encodeRecModeInfo(Resource report, Model dataModel) {
        Model resultModel = report.getModel();
        StringBuffer stringBufferQuery = PrefixesDel.getPrefixes(resultModel);
        String queryBody = "SELECT DISTINCT ?validationRs \n" +
                "WHERE{?s sh:conforms ?validationRs.}";

        stringBufferQuery.append(queryBody);
        Query querySPARQL = QueryFactory.create(stringBufferQuery.toString());

        // 执行查询并获取结果
        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL, resultModel)) {
            ResultSet rs = qexec.execSelect();

            if (rs.hasNext()) {
                String validationRs = rs.nextSolution().get("validationRs").toString();
                /*System.out.println(validationRs);*/

                if (validationRs.contains("false")) {  // 说明有索力传感器观测异常
                    generateJSONFile(resultModel, dataModel);
                } else {  // 说明所有索力传感器观测正常
                    // 创建一个JSONObject并添加数据
                    JSONObject jsonObject = new JSONObject();
                    jsonObject.put("message", "All the cable forces obtained by this monitoring is normal.");

                    // 将JSONObject写入文件
                    try (FileWriter file = new FileWriter("G:/KG_study/cableForce_reconstruction/config_files/no_reconstruction_requirement.json")) {
                        file.write(jsonObject.toString(4));  // 格式化输出，4代表缩进级别
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        // 创建一个空的 JSONObject
        JSONObject jsonObject = new JSONObject();

        // 将空的 JSON 对象写入文件
        try (FileWriter file = new FileWriter("G:/KG_study/cableForce_reconstruction/config_files/click.json")) {
            file.write(jsonObject.toString(4));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void generateJSONFile(Model resultModel, Model dataModel){
        StringBuffer stringBufferQuery01 = PrefixesDel.getPrefixes(dataModel);
        String queryBody01 = "SELECT ?envTempObs \n" +
                "WHERE{" +
                "?envTempObs a csm:EnvironmentalTemperatureObservation;" +
                "      csm:hasAbnormalObsRecentIndex ?abnormalIndex." +
                "      FILTER(?abnormalIndex > '-1'^^xsd:integer)}";

        stringBufferQuery01.append(queryBody01);
        Query querySPARQL01 = QueryFactory.create(stringBufferQuery01.toString());

        // 执行查询并获取结果
        ArrayList<String> hisAbnormalEnvTempObsTotal = new ArrayList<>();
        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL01, dataModel)) {
            ResultSet rs = qexec.execSelect();
            while (rs.hasNext()){
                String abnormalEnvTempObs = rs.nextSolution().get("envTempObs").toString();
                /*System.out.println(abnormalEnvTempObs);*/
                hisAbnormalEnvTempObsTotal.add(abnormalEnvTempObs);
            }
        }


        StringBuffer stringBufferQuery = PrefixesDel.getPrefixes(resultModel);
        String queryBody = "SELECT DISTINCT ?cableTensionObs ?recMode ?historicalAbnormalNum \n" +
                "WHERE{?s sh:focusNode ?cableTensionObs;" +
                "                sh:sourceShape ?recMode;" +
                "                sh:value   ?historicalAbnormalNum.}";

        stringBufferQuery.append(queryBody);
        Query querySPARQL = QueryFactory.create(stringBufferQuery.toString());

        Map<String, ArrayList<String>> dataMap = new HashMap<>();
        // 执行查询并获取结果
        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL, resultModel)) {
            ResultSet rs = qexec.execSelect();
            /*// 打印所有查询结果
            System.out.println("Query Results:");
            ResultSetFormatter.out(System.out, rs, querySPARQL);*/

            while (rs.hasNext()){
                QuerySolution solution = rs.nextSolution();
                String valueStr = solution.get("historicalAbnormalNum").toString();
                int index = valueStr.indexOf("^^");
                int abnormalNum = Integer.parseInt(valueStr.substring(0, index));
                if (abnormalNum == hisAbnormalEnvTempObsTotal.size()){
                    String keyJson = solution.get("cableTensionObs").toString();

                    // 使用构造函数复制 ArrayList
                    ArrayList<String> copiedList = new ArrayList<>(hisAbnormalEnvTempObsTotal);
                    copiedList.add(solution.get("recMode").toString());

                    dataMap.put(keyJson, copiedList);
                }else {
                    // 终止程序
                    System.out.println("诊断监测异常数目与查询不匹配，请重点核查！");
                    System.exit(0);

                }
            }

            // 创建 JSON 对象
            JSONObject jsonObject = new JSONObject();

            // 遍历 Map 并将键值对添加到 JSON 对象
            for (Map.Entry<String, ArrayList<String>> entry : dataMap.entrySet()) {
                JSONArray jsonArray = new JSONArray(entry.getValue());
                jsonObject.put(entry.getKey(), jsonArray);
            }

            // 将 JSON 对象写入文件
            try (FileWriter file = new FileWriter("G:/KG_study/cableForce_reconstruction/config_files/cableForce_rec_info.json")) {
                file.write(jsonObject.toString(4)); // 参数 4 表示缩进级别
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
