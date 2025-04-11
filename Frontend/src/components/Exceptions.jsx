// import React, { useState, useEffect } from 'react';
// import {
//   Table,
//   Card,
//   Button,
//   Modal,
//   Form,
//   Select,
//   DatePicker,
//   Tag,
//   Space,
//   message,
//   Input
// } from 'antd';
// import {
//   FilterOutlined,
//   ExceptionOutlined,
//   FileProtectOutlined
// } from '@ant-design/icons';
// import axios from 'axios';
// import { useForm, Controller } from 'react-hook-form';

// const { Option } = Select;
// const { RangePicker } = DatePicker;


// const Exceptions = () => {
//   const ResolveExceptionModal = ({
//     currentException,
//     isResolveModalVisible,
//     setIsResolveModalVisible,
//     documentFields,
//     onSubmit
//   }) => {
//     const {
//       handleSubmit,
//       control,
//       formState: { errors }
//     } = useForm({
//       defaultValues: {
//         resolutionDetails: currentException?.resolutionDetails || '',
//         ...Object.fromEntries(documentFields.map(f => [f.name, f.value]))
//       }
//     });
  
//     const handleResolveSubmit = async (data) => {
//       try {
//         const { resolutionDetails, ...fieldValues } = data;  
//         // Update the document using your new API endpoint format
//         const response = await axios.put(`http://localhost:3000/api/update_document`, {
//           document_id: currentException.documentId,
//           document_type: currentException.type.toLowerCase(), // Make sure type is lowercase to match backend
//           fields: fieldValues
//         });
    
//         message.success('Exception resolved and document updated successfully!');
//         setIsResolveModalVisible(false);
//         fetchExceptions();
//       } catch (err) {
//         console.error(err);
//         message.error('Failed to resolve and update document.');
//       }
//     };
  
//     return (
//       <Modal
//         title={`Resolve Exception - Document ID: ${currentException?.documentId}`}
//         open={isResolveModalVisible}
//         onCancel={() => setIsResolveModalVisible(false)}
//         footer={null}
//       >
//         <form onSubmit={handleSubmit((data) => handleResolveSubmit(data))}>
//           <div style={{ marginBottom: 16 }}>
//             <label> Resolution Details</label>
//             <Controller
//               name="resolutionDetails"
//               control={control}
//               rules={{ required: "Please input resolution details" }}
//               render={({ field }) => (
//                 <Input.TextArea
//                   {...field}
//                   rows={4}
//                   status={errors.resolutionDetails ? "error" : ""}
//                 />
//               )}
//             />
//             {errors.resolutionDetails && (
//               <p style={{ color: "red" }}>{errors.resolutionDetails.message}</p>
//             )}
//           </div>
  
//           {documentFields.map((field) => (
//             <div key={field.name} style={{ marginBottom: 16 }}>
//               <label><b>{field.name}</b> - {field.value? field.value : "Empty"}</label>
//               <Controller
//                 name={field.name}
//                 control={control}
//                 render={({ field: controllerField }) => (
//                   <Input {...controllerField}  />
//                 )}
//               />
//             </div>
//           ))}
  
//           <Button type="primary" htmlType="submit">
//             Resolve & Update
//           </Button>
//         </form>
//       </Modal>
//     );
//   };
//   const [isFilterModalVisible, setIsFilterModalVisible] = useState(false);
//   const [exceptions, setExceptions] = useState([]);
//   const [loading, setLoading] = useState(false);
//   const [filterForm] = Form.useForm();
//   const [isResolveModalVisible, setIsResolveModalVisible] = useState(false);
//   const [currentException, setCurrentException] = useState(null);
//   const [documentFields, setDocumentFields] = useState([]);

//   useEffect(() => {
//     fetchExceptions();
//   }, []);

//   const fetchExceptions = async (filters = {}) => {
//     setLoading(true);
//     try {
//       const response = await axios.get(`http://localhost:3000/api/exceptions`);
//       const raw = response.data.exceptions;

//       const formatted = raw.map((item, index) => ({
//         key: index,
//         documentId: item[0],
//         fileName: item[1],
//         type: item[2],
//         exceptionType: item[3],
//         batchId: item[4],
//         dateIdentified: item[5],
//         json: item[6]
//       }));

//       setExceptions(formatted);
//     } catch (error) {
//       console.error('Error fetching exceptions:', error);
//       message.error('Failed to load exceptions data');
//     } finally {
//       setLoading(false);
//     }
//   };

//   const handleResolveException = async (exceptionKey) => {
//     try {
//       const exception = exceptions[exceptionKey];
//       const docId = exception.documentId;
//       const docType = exception.type;

//       const response = await axios.post(`http://localhost:3000/api/exceptions_details`, {
//         document_id: docId
//       });
//       const data = response.data;

//       const dynamicFields = Object.entries(data.type_specific_data || {}).map(([key, value]) => ({
//         name: key,
//         value: value
//       }));

//       setDocumentFields(dynamicFields);

//       setCurrentException({
//         documentId: docId,
//         exceptionId: data.exceptionId,
//         type: docType,
//         resolutionDetails: data.resolutionDetails || '',
//       });

//       setIsResolveModalVisible(true);
//     } catch (error) {
//       console.error('Error fetching exception data:', error);
//       message.error('Failed to fetch exception details');
//     }
//   };

//   const handleResolveSubmit = async (data) => {
//     try {
//       const { resolutionDetails, ...fieldValues } = values;

//       await axios.post(`http://localhost:3000/api/exceptions/resolve`, {
//         exceptionId: currentException.exceptionId,
//         resolutionDetails
//       });

//       await axios.post(`http://localhost:3000/api/documents/update`, {
//         documentId: currentException.documentId,
//         type: currentException.type,
//         fields: fieldValues
//       });

//       message.success('Exception resolved and document updated successfully!');
//       setIsResolveModalVisible(false);
//       fetchExceptions();
//     } catch (err) {
//       console.error(err);
//       message.error('Failed to resolve and update document.');
//     }
//   };

//   const handleViewException = (exception) => {
//     Modal.info({
//       title: `Exception Details: ${exception.documentId}`,
//       content: (
//         <div>
//           <p>File Name: {exception.fileName}</p>
//           <p>Document Type: {exception.type}</p>
//           <p>Exception Type: {exception.exceptionType}</p>
//           <p>Batch ID: {exception.batchId}</p>
//           <p>Date Identified: {exception.dateIdentified}</p>
//         </div>
//       ),
//       onOk() { },
//     });
//   };

//   const columns = [
//     {
//       title: 'Document ID',
//       dataIndex: 'documentId',
//       key: 'documentId',
//     },
//     {
//       title: 'File Name',
//       dataIndex: 'fileName',
//       key: 'fileName',
//     },
//     {
//       title: 'Type',
//       dataIndex: 'type',
//       key: 'type',
//       filters: [
//         { text: 'Claim', value: 'Claim' },
//         { text: 'Agency', value: 'Agency' },
//         { text: 'Mortgage', value: 'Mortgage' },
//       ],
//       onFilter: (value, record) => record.type === value,
//     },
//     {
//       title: 'Exception Type',
//       dataIndex: 'exceptionType',
//       key: 'exceptionType',
//       render: (exceptionType) => {
//         const color = {
//           'Missing Data': 'orange',
//           'Validation Error': 'red',
//           'Format Issue': 'volcano'
//         }[exceptionType];
//         return <Tag color={color}>{exceptionType}</Tag>;
//       }
//     },
//     {
//       title: 'Batch ID',
//       dataIndex: 'batchId',
//       key: 'batchId',
//     },
//     {
//       title: 'Date Identified',
//       dataIndex: 'dateIdentified',
//       key: 'dateIdentified',
//     },
//     {
//       title: 'Actions',
//       key: 'actions',
//       render: (_, record) => (
//         <Space>
//           <Button
//             size="small"
//             icon={<ExceptionOutlined />}
//             onClick={() => handleViewException(record)}
//           >
//             Details
//           </Button>
//           <Button
//             size="small"
//             icon={<FileProtectOutlined />}
//             type="primary"
//             onClick={() => handleResolveException(record.key)}
//           >
//             Resolve
//           </Button>
//         </Space>
//       )
//     }
//   ];

//   const handleFilterSubmit = (values) => {
//     fetchExceptions(values);
//     setIsFilterModalVisible(false);
//   };

//   const handleResetFilters = () => {
//     filterForm.resetFields();
//   };

//   return (
//     <Card
//       title="Exceptions"
//       extra={
//         <Button
//           icon={<FilterOutlined />}
//           onClick={() => setIsFilterModalVisible(true)}
//         >
//           Filters
//         </Button>
//       }
//     >
//       <Table
//         columns={columns}
//         dataSource={exceptions}
//         loading={loading}
//         pagination={{ pageSize: 5 }}
//         rowKey="key"
//       />

//       <ResolveExceptionModal
//         currentException={currentException}
//         isResolveModalVisible={isResolveModalVisible}
//         setIsResolveModalVisible={setIsResolveModalVisible}
//         documentFields={documentFields}
//         onSubmit={handleResolveSubmit}
//       />

//       <Modal
//         title="Filter Exceptions"
//         open={isFilterModalVisible}
//         onCancel={() => setIsFilterModalVisible(false)}
//         footer={null}
//       >
//         <Form
//           form={filterForm}
//           onFinish={handleFilterSubmit}
//           layout="vertical"
//         >
//           <Form.Item name="documentType" label="Document Type">
//             <Select placeholder="Select document type" allowClear>
//               <Option value="Claim">Claim</Option>
//               <Option value="Agency">Agency</Option>
//               <Option value="Mortgage">Mortgage</Option>
//             </Select>
//           </Form.Item>
//           <Form.Item name="exceptionType" label="Exception Type">
//             <Select placeholder="Select exception type" allowClear>
//               <Option value="Missing Data">Missing Data</Option>
//               <Option value="Validation Error">Validation Error</Option>
//               <Option value="Format Issue">Format Issue</Option>
//             </Select>
//           </Form.Item>
//           <Form.Item name="dateIdentifiedRange" label="Date Identified Range">
//             <RangePicker />
//           </Form.Item>
//           <Form.Item>
//             <Space>
//               <Button type="primary" htmlType="submit">
//                 Apply Filters
//               </Button>
//               <Button onClick={handleResetFilters}>
//                 Reset
//               </Button>
//             </Space>
//           </Form.Item>
//         </Form>
//       </Modal>
//     </Card>
//   );
// };

// export default Exceptions;


import React, { useState, useEffect } from 'react';
import {
  Table,
  Card,
  Button,
  Modal,
  Form,
  Select,
  DatePicker,
  Tag,
  Space,
  message,
  Input,
  Tabs
} from 'antd';
import {
  FilterOutlined,
  ExceptionOutlined,
  FileProtectOutlined,
  CodeOutlined,
  SaveOutlined
} from '@ant-design/icons';
import axios from 'axios';
import { useForm, Controller } from 'react-hook-form';
import ReactJson from 'react-json-view';

const { Option } = Select;
const { RangePicker } = DatePicker;
const { TabPane } = Tabs;

const Exceptions = () => {
  const ResolveExceptionModal = ({
    currentException,
    isResolveModalVisible,
    setIsResolveModalVisible,
    documentFields,
    onSubmit
  }) => {
    const {
      handleSubmit,
      control,
      formState: { errors }
    } = useForm({
      defaultValues: {
        resolutionDetails: currentException?.resolutionDetails || '',
        ...Object.fromEntries(documentFields.map(f => [f.name, f.value]))
      }
    });
  
    const handleResolveSubmit = async (data) => {
      try {
        const { resolutionDetails, ...fieldValues } = data;  
        // Update the document using your new API endpoint format
        const response = await axios.put(`http://localhost:3000/api/update_document`, {
          document_id: currentException.documentId,
          document_type: currentException.type.toLowerCase(), // Make sure type is lowercase to match backend
          fields: fieldValues
        });
    
        message.success('Exception resolved and document updated successfully!');
        setIsResolveModalVisible(false);
        fetchExceptions();
      } catch (err) {
        console.error(err);
        message.error('Failed to resolve and update document.');
      }
    };
  
    return (
      <Modal
        title={`Resolve Exception - Document ID: ${currentException?.documentId}`}
        open={isResolveModalVisible}
        onCancel={() => setIsResolveModalVisible(false)}
        footer={null}
      >
        <form onSubmit={handleSubmit((data) => handleResolveSubmit(data))}>
          <div style={{ marginBottom: 16 }}>
            <label> Resolution Details</label>
            <Controller
              name="resolutionDetails"
              control={control}
              rules={{ required: "Please input resolution details" }}
              render={({ field }) => (
                <Input.TextArea
                  {...field}
                  rows={4}
                  status={errors.resolutionDetails ? "error" : ""}
                />
              )}
            />
            {errors.resolutionDetails && (
              <p style={{ color: "red" }}>{errors.resolutionDetails.message}</p>
            )}
          </div>
  
          {documentFields.map((field) => (
            <div key={field.name} style={{ marginBottom: 16 }}>
              <label><b>{field.name}</b> - {field.value? field.value : "Empty"}</label>
              <Controller
                name={field.name}
                control={control}
                render={({ field: controllerField }) => (
                  <Input {...controllerField}  />
                )}
              />
            </div>
          ))}
  
          <Button type="primary" htmlType="submit">
            Resolve & Update
          </Button>
        </form>
      </Modal>
    );
  };

  // JSON Editor Modal Component
  const JsonEditorModal = ({
    isVisible,
    setIsVisible,
    documentId,
    jsonData,
    title
  }) => {
    const [editedJson, setEditedJson] = useState(jsonData);
    const [isSaving, setIsSaving] = useState(false);

    const handleJsonEdit = (edit) => {
      setEditedJson(edit.updated_src);
    };

    const saveJsonChanges = async () => {
      if (!documentId || !editedJson) return;
      
      setIsSaving(true);
      try {
        // Make API call to save the updated JSON
        await axios.post("http://localhost:3000/api/update_json", {
          document_id: documentId,
          json_content: editedJson
        });
        
        message.success("JSON updated successfully");
        setIsVisible(false);
        fetchExceptions(); // Refresh exceptions list
      } catch (error) {
        console.error("Error updating JSON:", error);
        message.error("Failed to update JSON data");
      } finally {
        setIsSaving(false);
      }
    };

    // Function to extract and display key information from JSON
    const renderKeyInfo = (json) => {
      if (!json || !json.Important_Info) return <p>No important information available</p>;
      
      const info = json.Important_Info;
      const entries = Object.entries(info);
      
      return (
        <ul style={{ listStyleType: 'none', padding: 0 }}>
          {entries.map(([key, value]) => (
            <li key={key} style={{ marginBottom: '8px' }}>
              <strong>{key}:</strong> {value}
            </li>
          ))}
        </ul>
      );
    };

    return (
      <Modal
        title={
          <div style={{ display: "flex", alignItems: "center" }}>
            <CodeOutlined style={{ marginRight: 8 }} />
            <span>{title || "Document JSON"}</span>
          </div>
        }
        open={isVisible}
        onCancel={() => setIsVisible(false)}
        footer={[
          <Button 
            key="save" 
            type="primary" 
            icon={<SaveOutlined />} 
            onClick={saveJsonChanges}
            loading={isSaving}
            disabled={!editedJson}
            style={{ marginRight: 8 }}
          >
            Save Changes
          </Button>,
          <Button key="close" onClick={() => setIsVisible(false)}>
            Close
          </Button>,
        ]}
        width={800}
      >
        {editedJson ? (
          <Tabs defaultActiveKey="keyInfo">
            <TabPane tab="Key Information" key="keyInfo">
              {renderKeyInfo(editedJson)}
            </TabPane>
            <TabPane tab="Edit JSON" key="editJson">
              <div style={{ backgroundColor: "#f5f5f5", padding: 16, borderRadius: 4, maxHeight: "400px", overflow: "auto" }}>
                <ReactJson
                  src={editedJson}
                  theme="rjv-default"
                  displayDataTypes={false}
                  name={false}
                  collapsed={1}
                  enableClipboard={true}
                  onEdit={handleJsonEdit}
                  onAdd={handleJsonEdit}
                  onDelete={handleJsonEdit}
                />
              </div>
            </TabPane>
          </Tabs>
        ) : (
          <p>No JSON data available for this document</p>
        )}
      </Modal>
    );
  };

  const [isFilterModalVisible, setIsFilterModalVisible] = useState(false);
  const [exceptions, setExceptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filterForm] = Form.useForm();
  const [isResolveModalVisible, setIsResolveModalVisible] = useState(false);
  const [currentException, setCurrentException] = useState(null);
  const [documentFields, setDocumentFields] = useState([]);
  
  // New states for JSON editing
  const [isJsonModalVisible, setIsJsonModalVisible] = useState(false);
  const [currentJsonData, setCurrentJsonData] = useState(null);
  const [currentDocumentId, setCurrentDocumentId] = useState(null);
  const [currentFileName, setCurrentFileName] = useState('');

  useEffect(() => {
    fetchExceptions();
  }, []);

  const fetchExceptions = async (filters = {}) => {
    setLoading(true);
    try {
      const response = await axios.get(`http://localhost:3000/api/exceptions`);
      const raw = response.data.exceptions;

      const formatted = raw.map((item, index) => ({
        key: index,
        documentId: item[0],
        fileName: item[1],
        type: item[2],
        exceptionType: item[3],
        batchId: item[4],
        dateIdentified: item[5],
        jsonContent: item[6] ? JSON.parse(item[6]) : null, // Parse JSON content if available
      }));

      setExceptions(formatted);
    } catch (error) {
      console.error('Error fetching exceptions:', error);
      message.error('Failed to load exceptions data');
    } finally {
      setLoading(false);
    }
  };

  const handleResolveException = async (exceptionKey) => {
    try {
      const exception = exceptions[exceptionKey];
      const docId = exception.documentId;
      const docType = exception.type;

      const response = await axios.post(`http://localhost:3000/api/exceptions_details`, {
        document_id: docId
      });
      const data = response.data;

      const dynamicFields = Object.entries(data.type_specific_data || {}).map(([key, value]) => ({
        name: key,
        value: value
      }));

      setDocumentFields(dynamicFields);

      setCurrentException({
        documentId: docId,
        exceptionId: data.exceptionId,
        type: docType,
        resolutionDetails: data.resolutionDetails || '',
      });

      setIsResolveModalVisible(true);
    } catch (error) {
      console.error('Error fetching exception data:', error);
      message.error('Failed to fetch exception details');
    }
  };

  // New function to handle JSON editing
  const handleEditJson = async (exceptionKey) => {
    try {
      const exception = exceptions[exceptionKey];
      console.log(exception)
      if (!exception.jsonContent) {
        // If no JSON content in the exception record, try to fetch it
        const response = await axios.post(`http://localhost:3000/api/get_document_json`, {
          document_id: exception.documentId
        });
        
        if (response.data && response.data.json_content) {
          setCurrentJsonData(response.data.json_content);
        } else {
          // If still no JSON content, show error
          message.warning('No JSON data available for this document');
          return;
        }
      } else {
        // Use the JSON content from the exception record
        setCurrentJsonData(exception.jsonContent);
      }
      
      setCurrentDocumentId(exception.documentId);
      setCurrentFileName(exception.fileName);
      setIsJsonModalVisible(true);
    } catch (error) {
      console.error('Error fetching document JSON:', error);
      message.error('Failed to fetch document JSON data');
    }
  };

  const handleResolveSubmit = async (data) => {
    try {
      const { resolutionDetails, ...fieldValues } = values;

      await axios.post(`http://localhost:3000/api/exceptions/resolve`, {
        exceptionId: currentException.exceptionId,
        resolutionDetails
      });

      await axios.post(`http://localhost:3000/api/documents/update`, {
        documentId: currentException.documentId,
        type: currentException.type,
        fields: fieldValues
      });

      message.success('Exception resolved and document updated successfully!');
      setIsResolveModalVisible(false);
      fetchExceptions();
    } catch (err) {
      console.error(err);
      message.error('Failed to resolve and update document.');
    }
  };

  const handleViewException = (exception) => {
    Modal.info({
      title: `Exception Details: ${exception.documentId}`,
      content: (
        <div>
          <p>File Name: {exception.fileName}</p>
          <p>Document Type: {exception.type}</p>
          <p>Exception Type: {exception.exceptionType}</p>
          <p>Batch ID: {exception.batchId}</p>
          <p>Date Identified: {exception.dateIdentified}</p>
        </div>
      ),
      onOk() { },
    });
  };

  const columns = [
    {
      title: 'Document ID',
      dataIndex: 'documentId',
      key: 'documentId',
    },
    {
      title: 'File Name',
      dataIndex: 'fileName',
      key: 'fileName',
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      filters: [
        { text: 'Claim', value: 'Claim' },
        { text: 'Agency', value: 'Agency' },
        { text: 'Mortgage', value: 'Mortgage' },
      ],
      onFilter: (value, record) => record.type === value,
    },
    {
      title: 'Exception Type',
      dataIndex: 'exceptionType',
      key: 'exceptionType',
      render: (exceptionType) => {
        const color = {
          'Missing Data': 'orange',
          'Validation Error': 'red',
          'Format Issue': 'volcano'
        }[exceptionType];
        return <Tag color={color}>{exceptionType}</Tag>;
      }
    },
    {
      title: 'Batch ID',
      dataIndex: 'batchId',
      key: 'batchId',
    },
    {
      title: 'Date Identified',
      dataIndex: 'dateIdentified',
      key: 'dateIdentified',
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            size="small"
            icon={<ExceptionOutlined />}
            onClick={() => handleViewException(record)}
          >
            Details
          </Button>
          <Button
            size="small"
            icon={<CodeOutlined />}
            onClick={() => handleEditJson(record.key)}
          >
            Edit JSON
          </Button>
          <Button
            size="small"
            icon={<FileProtectOutlined />}
            type="primary"
            onClick={() => handleResolveException(record.key)}
          >
            Resolve
          </Button>
        </Space>
      )
    }
  ];

  const handleFilterSubmit = (values) => {
    fetchExceptions(values);
    setIsFilterModalVisible(false);
  };

  const handleResetFilters = () => {
    filterForm.resetFields();
  };

  return (
    <Card
      title="Exceptions"
      extra={
        <Button
          icon={<FilterOutlined />}
          onClick={() => setIsFilterModalVisible(true)}
        >
          Filters
        </Button>
      }
    >
      <Table
        columns={columns}
        dataSource={exceptions}
        loading={loading}
        pagination={{ pageSize: 5 }}
        rowKey="key"
      />

      {/* Resolve Exception Modal */}
      <ResolveExceptionModal
        currentException={currentException}
        isResolveModalVisible={isResolveModalVisible}
        setIsResolveModalVisible={setIsResolveModalVisible}
        documentFields={documentFields}
        onSubmit={handleResolveSubmit}
      />

      {/* JSON Editor Modal */}
      <JsonEditorModal
        isVisible={isJsonModalVisible}
        setIsVisible={setIsJsonModalVisible}
        documentId={currentDocumentId}
        jsonData={currentJsonData}
        title={`Edit JSON - ${currentFileName}`}
      />

      {/* Filter Modal */}
      <Modal
        title="Filter Exceptions"
        open={isFilterModalVisible}
        onCancel={() => setIsFilterModalVisible(false)}
        footer={null}
      >
        <Form
          form={filterForm}
          onFinish={handleFilterSubmit}
          layout="vertical"
        >
          <Form.Item name="documentType" label="Document Type">
            <Select placeholder="Select document type" allowClear>
              <Option value="Claim">Claim</Option>
              <Option value="Agency">Agency</Option>
              <Option value="Mortgage">Mortgage</Option>
            </Select>
          </Form.Item>
          <Form.Item name="exceptionType" label="Exception Type">
            <Select placeholder="Select exception type" allowClear>
              <Option value="Missing Data">Missing Data</Option>
              <Option value="Validation Error">Validation Error</Option>
              <Option value="Format Issue">Format Issue</Option>
            </Select>
          </Form.Item>
          <Form.Item name="dateIdentifiedRange" label="Date Identified Range">
            <RangePicker />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Apply Filters
              </Button>
              <Button onClick={handleResetFilters}>
                Reset
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

export default Exceptions;