import React, { useEffect, useState } from 'react';
import { 
  Table, 
  Card, 
  Button, 
  Modal, 
  Form, 
  Select, 
  DatePicker, 
  Space,
  Tag,
  Spin,
  message
} from 'antd';
import axios from 'axios';
import { 
  FilterOutlined, 
  EyeOutlined, 
  FileTextOutlined 
} from '@ant-design/icons';

const { Option } = Select;
const { RangePicker } = DatePicker;

const BatchManagement = () => {
  const [isFilterModalVisible, setIsFilterModalVisible] = useState(false);
  const [batches, setBatches] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchBatches = async () => {
    try {
      const response = await axios.get('http://localhost:3000/api/batches');
      const data = response.data.batches.map((item, index) => ({
        key: item[0],
        batchId: item[1],
        date: new Date(item[2]).toLocaleDateString(),  // Format as needed
        totalDocuments: item[3],
        exceptions: item[4],
        status: item[4] > 0 ? 'Exceptions' : 'Complete' // You can adjust logic here
      }));
      setBatches(data);
    } catch (err) {
      console.error(err);
      message.error("Failed to load batches");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBatches();
  }, []);

  const columns = [
    {
      title: 'Batch ID',
      dataIndex: 'batchId',
      key: 'batchId',
    },
    {
      title: 'Total Documents',
      dataIndex: 'totalDocuments',
      key: 'totalDocuments',
    },
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
    },
    {
      title: 'Exceptions',
      dataIndex: 'exceptions',
      key: 'exceptions',
      render: (exceptions) => exceptions > 0 ? 
        <Tag color="red">{exceptions} Exceptions</Tag> : 
        <Tag color="green">No Exceptions</Tag>
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => handleViewBatch(record)}
          >
            View
          </Button>
          <Button 
            size="small" 
            icon={<FileTextOutlined />}
            type="primary"
          >
            Details
          </Button>
        </Space>
      )
    }
  ];

  const handleFilterSubmit = (values) => {
    console.log('Filter values:', values);
    setIsFilterModalVisible(false);
  };

  const handleViewBatch = (batch) => {
    Modal.info({
      title: `Batch Details: ${batch.batchId}`,
      content: (
        <div>
          <p>Total Documents: {batch.totalDocuments}</p>
          <p>Status: {batch.status}</p>
          <p>Date: {batch.date}</p>
          <p>Exceptions: {batch.exceptions}</p>
        </div>
      ),
      onOk() {},
    });
  };

  return (
    <Card 
      title="Batch Management" 
      extra={
        <Button 
          icon={<FilterOutlined />} 
          onClick={() => setIsFilterModalVisible(true)}
        >
          Filters
        </Button>
      }
    >
      {loading ? (
        <Spin />
      ) : (
        <Table 
          columns={columns} 
          dataSource={batches}
          pagination={{ pageSize: 5 }}
        />
      )}

      <Modal
        title="Filter Batches"
        open={isFilterModalVisible}
        onCancel={() => setIsFilterModalVisible(false)}
        footer={null}
      >
        <Form onFinish={handleFilterSubmit}>
          <Form.Item name="status" label="Status">
            <Select placeholder="Select batch status">
              <Option value="Complete">Complete</Option>
              <Option value="Processing">Processing</Option>
              <Option value="Exceptions">Exceptions</Option>
            </Select>
          </Form.Item>
          <Form.Item name="dateRange" label="Date Range">
            <RangePicker />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">
              Apply Filters
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

export default BatchManagement;
