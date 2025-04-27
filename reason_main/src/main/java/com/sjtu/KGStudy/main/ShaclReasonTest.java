package com.sjtu.KGStudy.main;

import com.sjtu.KGStudy.utils.*;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Resource;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.jena.rdf.model.Model;
import org.apache.jena.util.FileUtils;
import org.topbraid.jenax.util.JenaUtil;
import org.topbraid.shacl.util.ModelPrinter;
import org.topbraid.shacl.validation.ValidationUtil;


public class ShaclReasonTest {
    public static void main(String[] args) {
        String monitorDataPath = "G:/KG_study/reason_main/src/main/resources/monitorData.json";
        Map<String, Object[]> monitorDataMap = getMonitorData(monitorDataPath);

        Model dataModel = JenaUtil.createMemoryModel();
        Model shapesModelMonitorAnomaly = JenaUtil.createMemoryModel();

        String csmFilepath = "G:/KG_study/reason_main/src/main/resources/myDataGraph/CableStructureMonitor829.ttl";

        // dataModel就是被检测的模型
        try {
            InputStream inDataModel = new FileInputStream(csmFilepath);
            dataModel.read(inDataModel, "urn:dummy", FileUtils.langTurtle);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        shapesModelMonitorAnomaly.read(ShaclReasonTest.class.getResourceAsStream("/myShapesGraph/monitorAnomaly.ttl"), "urn:dummy", FileUtils.langTurtle);

        // 需要记录要执行数据重构的监测数据类型
        Map<String, String[]> dataReconstructionInfo = new HashMap<>();
        for (Map.Entry<String, Object[]> entry : monitorDataMap.entrySet()){
            // key = "cableTemperature9";

            String key = entry.getKey();
            String monitorTime = (String) entry.getValue()[0];

            double monitorVaule = (Double) entry.getValue()[1];

            // 定义正则表达式
            Pattern pattern = Pattern.compile("(\\D+)(\\d+)");
            Matcher matcher = pattern.matcher(key);

            String textPart = "";
            String numberPart = "";
            if (matcher.matches()) {
                textPart = matcher.group(1);
                numberPart = matcher.group(2);
            } else {
                System.out.println("No match found.");
                System.exit(1);
            }

            String subjectStr = "inst:" + textPart + "Obs" + numberPart;

            // 动态更新数据模型
            RDFModelUpdateTools.updateMonitorTime(dataModel, subjectStr, monitorTime);
            RDFModelUpdateTools.updateMonitorData(dataModel, subjectStr, monitorVaule);
            /*QuerySPARQLUtils.queryMonitorData(dataModel, predicate);*/

            // 获取阈值并动态更新形状图
            float[] thresholdList = ThresholdCalculation.calculateThresholds();
            RDFModelUpdateTools.updateShapesGraphMonitorData(shapesModelMonitorAnomaly, subjectStr, thresholdList[0], thresholdList[1]);

            // 基于SHACL的监测数据异常诊断
            Resource reportMonitorAnomaly = ValidationUtil.validateModel(dataModel, shapesModelMonitorAnomaly, true);
            // System.out.println(ModelPrinter.get().print(reportMonitorAnomaly.getModel()));

            Object[] reasonResults = analysisMonitorAnomalyReasonReport(reportMonitorAnomaly);
            String focusNode = (String) reasonResults[0];
            boolean labelOption = (boolean) reasonResults[1];

            if (labelOption){  // 如果是异常数据，则必须将当前观测状态标记为false，意为监测异常
                DelSHACLReasonResultMonitorData.labelObservationStatus(dataModel, subjectStr, false);
            }else {  //如果不是异常数据，就必须利用SPARQL查到所校验的传感器，这样才能保证后续程序正常执行
                DelSHACLReasonResultMonitorData.labelObservationStatus(dataModel, subjectStr, true);
                focusNode = QuerySPARQLUtils.getTargetSensorURI(dataModel, subjectStr);
            }

            // 基于诊断异常的传感器异常数据标记以及数据重构
            // 由于OWL有开放世界假设，因此当异常超过10时，插入新的异常标记也不会报错，反而一旦数据正常以后，就会立刻重置进而正常使用
            DelSHACLReasonResultMonitorData.setMonitorDataAnomaly(dataModel, focusNode, labelOption);

            if (labelOption){  // 需要启动数据重构代理模型来重构此类型数据
                String[] monitorInfo = DelSHACLReasonResultMonitorData.getMonitorTargetAndType(dataModel, focusNode);

                dataReconstructionInfo.put(key, monitorInfo);  // 将异常监测信息存入Map以便后续导出
            }

            // break;
        }

        // 将需要重构数据的类型导出本地供python代码识别
        if (! dataReconstructionInfo.isEmpty()){
            // 将Map转换为JSONObject
            JSONObject jsonObject = new JSONObject(dataReconstructionInfo);

            // 将JSONObject写入本地文件
            try (FileWriter fileWriter = new FileWriter("G:/KG_study/data_reconstruction/config_file/data_reconstruction_type.json")) {
                fileWriter.write(jsonObject.toString(4)); // 格式化输出，缩进4个空格
            } catch (IOException e) {
                e.printStackTrace();
            }

            // 对于拉索索力重构，根据监测数据缺失情况，开发了两种重构模式，现在基于知识推理来选择其中一种模式
            /*
             * 模式1：索力监测异常而索温度监测正常；模式2：索力监测异常并且索温度监测异常
             **/
            Model shapesModelCableTensionRecMode = JenaUtil.createMemoryModel();
            shapesModelCableTensionRecMode.read(ShaclReasonTest.class.getResourceAsStream("/myShapesGraph/cableTensionReconstructionMode.ttl"), "urn:dummy", FileUtils.langTurtle);
            Resource reportCableTensionRecMode = ValidationUtil.validateModel(dataModel, shapesModelCableTensionRecMode, true);  // 基于SHACL的监测数据异常诊断
            // System.out.println(ModelPrinter.get().print(reportCableTensionRecMode.getModel()));
            Map<String, String> cableTensionRecInfo = analysisCableTensionRecModeReasonReport(reportCableTensionRecMode);
            JSONObject jsonObject02 = new JSONObject(cableTensionRecInfo);  // 将Map转换为JSONObject
            try (FileWriter fileWriter = new FileWriter("G:/KG_study/data_reconstruction/config_file/cable_tension_rec_mode.json")) {  // 将JSONObject写入本地文件
                fileWriter.write(jsonObject02.toString(4)); // 格式化输出，缩进4个空格
            } catch (IOException e) {
                e.printStackTrace();
            }

            // 准备一个用于触发python程序自动执行的click文件
            String clickFilePath = "G:/KG_study/data_reconstruction/config_file/click.json";
            File jsonFile = new File(clickFilePath);

            // 判断文件是否存在
            if (jsonFile.exists()) {
                // 文件存在，删除文件
                if (! jsonFile.delete()) {
                    System.out.println("文件删除失败: " + clickFilePath);
                }
            }

            // 创建新文件
            try {
                boolean clickCreation = jsonFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // 已经执行完该时间节点所有传感器异常数据标记，现在对修订模型执行异常状态传感器识别
        /*
         * 首先创建形状图，然后确定异常工作状态传感器，并将传感器状态标记为异常；最后再解析工作报告从IFCOWL中确定拉索编号及拉索位置
         *
         * 要不要加一个有趣的检测服务，当相邻拉索（索力重构的输入）有一个发生传感器损坏时，提升传感器修理的紧迫性？-->先不予考虑，形状图太蠢了！
         */
        Model shapesModelSensorAnomaly = JenaUtil.createMemoryModel();
        shapesModelSensorAnomaly.read(ShaclReasonTest.class.getResourceAsStream("/myShapesGraph/sensorStatus.ttl"), "urn:dummy", FileUtils.langTurtle);
        Resource reportSensorAnomaly = ValidationUtil.validateModel(dataModel, shapesModelSensorAnomaly, true);  // 基于SHACL的传感器异常诊断
        // System.out.println(ModelPrinter.get().print(reportSensorAnomaly.getModel()));
        ArrayList<String> abnormalSensorURIs = analysisSensorAnomalyReasonReport(reportSensorAnomaly);  // 分析推理报告确定异常传感器

        if (abnormalSensorURIs.size() != 0){
            // 创建由IFC模型转化而来的RDF模型，可以是IFCOWL驱动的，也可以是LBD或其他；这里以IFCOWL为例
            Model bimConversionModel = JenaUtil.createMemoryModel();
            bimConversionModel.read(ShaclReasonTest.class.getResourceAsStream("/myDataGraph/useCableClamp_IFC2RDF.ttl"), "urn:dummy", FileUtils.langTurtle);

            for (String abnormalSensorURI : abnormalSensorURIs){  //依次处理异常传感器，并发出去修理哪根索上传感器的建议
                System.out.println("\n" + "开始核查异常传感器" + abnormalSensorURI + "的基本信息。");
                DelSHACLReasonResultAbnormalSensor.labelAbnormalSensor(dataModel, abnormalSensorURI);
                DelSHACLReasonResultAbnormalSensor.makeDecisionSensorRepair(dataModel, bimConversionModel, abnormalSensorURI);

                if (abnormalSensorURI.contains("cableTension")){  // 对拉索传感器执行一个比较有意思的评估，即看维修的紧迫程度
                    DelSHACLReasonResultAbnormalSensor.reportCableTensionRepairUrgency(dataModel, abnormalSensorURI);
                }
            }
        }

        // 存储监测数据异常标记后的数据模型
        try (OutputStream out = new FileOutputStream(csmFilepath)) {  // 被迫使用绝对路径
            dataModel.write(out, "TURTLE");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Map<String, Object[]> getMonitorData(String monitorDataPath){
        // 将JSONObject转换为Map
        Map<String, Object[]> monitorDataMap = new HashMap<>();

        try (FileReader fileReader = new FileReader(monitorDataPath)) {
            int ch;
            StringBuilder sb = new StringBuilder();
            while ((ch = fileReader.read()) != -1) {
                sb.append((char) ch);
            }
            JSONObject readJsonObject = new JSONObject(sb.toString());

            Iterator<String> keys = readJsonObject.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                JSONArray jsonArray = readJsonObject.getJSONArray(key);
                String strValue = jsonArray.getString(0);
                double doubleValue = jsonArray.getDouble(1);
                monitorDataMap.put(key, new Object[]{strValue, doubleValue});
            }

            /*for (Map.Entry<String, Object[]> entry : monitorDataMap.entrySet()) {
                System.out.println("Key: " + entry.getKey() + ", Value: " + entry.getValue()[0] + ", " + entry.getValue()[1]);
            }*/
        } catch (IOException e) {
            e.printStackTrace();
        }

        return monitorDataMap;
    }

    public static Object[] analysisMonitorAnomalyReasonReport(Resource report){
        Object[] results = new Object[2];

        Model resultModel = report.getModel();

        StringBuffer stringBufferQuery = PrefixesDel.getPrefixes(resultModel);
        String queryBody = "SELECT ?focusNode \n" +
                "WHERE{?s sh:focusNode ?focusNode.}";

        stringBufferQuery.append(queryBody);
        Query querySPARQL = QueryFactory.create(stringBufferQuery.toString());

        // 执行查询并获取结果
        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL, resultModel)) {
            ResultSet rs = qexec.execSelect();

            if (rs.hasNext()){  // 这里最多只有一个查询结果
                String focusNodeFullPre = rs.nextSolution().get("focusNode").toString();
                results[0] = PrefixesDel.replacePrefix(resultModel, focusNodeFullPre);

                results[1] = true;  // 如果rs有查询结果，则说明获得了异常监测数据
            }else{
                results[0] = "";
                results[1] = false;
            }
        }

        return results;
    }

    public static Map<String, String> analysisCableTensionRecModeReasonReport(Resource report){
        Map<String, String> cableTensionRecInfo = new HashMap<>();

        Model resultModel = report.getModel();

        StringBuffer stringBufferQuery = PrefixesDel.getPrefixes(resultModel);
        String queryBody = "SELECT ?focusNode ?recModeShape \n" +
                "WHERE{?s sh:resultSeverity sh:Violation;"
                + "       sh:focusNode ?focusNode;"
                + "       sh:sourceShape ?recModeShape.}";

        stringBufferQuery.append(queryBody);
        Query querySPARQL = QueryFactory.create(stringBufferQuery.toString());

        // 执行查询并获取结果
        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL, resultModel)) {
            ResultSet rs = qexec.execSelect();

            String key;
            String valueFullStr;
            int position;
            String value;
            while (rs.hasNext()){
                QuerySolution solutionSPQ = rs.nextSolution();
                key = PrefixesDel.replacePrefix(resultModel, solutionSPQ.get("focusNode").toString());
                valueFullStr = solutionSPQ.get("recModeShape").toString();
                position = valueFullStr.indexOf("#");
                value = valueFullStr.substring(position+1, valueFullStr.length()-5);

                cableTensionRecInfo.put(key, value);
            }

        }

        return cableTensionRecInfo;
    }

    public static ArrayList<String> analysisSensorAnomalyReasonReport(Resource report){
        ArrayList<String> abnormalSensorURIs = new ArrayList<>();

        Model resultModel = report.getModel();

        StringBuffer stringBufferQuery = PrefixesDel.getPrefixes(resultModel);
        String queryBody = "SELECT ?focusNode \n" +
                "WHERE{?s sh:resultSeverity sh:Violation;"
                + "       sh:focusNode ?focusNode.}";

        stringBufferQuery.append(queryBody);
        Query querySPARQL = QueryFactory.create(stringBufferQuery.toString());

        // 执行查询并获取结果
        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL, resultModel)) {
            ResultSet rs = qexec.execSelect();

            while (rs.hasNext()){
                String focusNodeFullPre = rs.nextSolution().get("focusNode").toString();
                String simplifiedURI = PrefixesDel.replacePrefix(resultModel, focusNodeFullPre);
                /*// ANSI转义码
                String redColor = "\u001B[31m";
                String resetColor = "\u001B[0m";
                String boldText = "\u001B[1m";

                // 打印红色加粗的警告信息
                System.out.println(boldText + redColor + "Sensor labeled as " + simplifiedURI + " need to be repaired!" + resetColor);*/
                abnormalSensorURIs.add(simplifiedURI);
            }

        }

        return abnormalSensorURIs;
    }
}
