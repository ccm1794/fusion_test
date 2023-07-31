        if(box.y>=0)
        {
          points_1.x = box.x - (box.size_x)/2;
          points_1.y = box.y + (box.size_y)/2;
          points_1.z = box.z - (box.size_z)/2;

          points_2.x = box.x + (box.size_x)/2;
          points_2.y = box.y - (box.size_y)/2;
          points_2.z = box.z - (box.size_z)/2;

          points_3.x = box.x + (box.size_x)/2;
          points_3.y = box.y - (box.size_y)/2;
          points_3.z = box.z + (box.size_z)/2;

          points_4.x = box.x - (box.size_x)/2;
          points_4.y = box.y + (box.size_y)/2;
          points_4.z = box.z + (box.size_z)/2;
        }
        else if(box.y<0)
        {
          points_1.x = box.x - (box.size_x)/2;
          points_1.y = box.y - (box.size_y)/2;
          points_1.z = box.z - (box.size_z)/2;

          points_2.x = box.x + (box.size_x)/2;
          points_2.y = box.y + (box.size_y)/2;
          points_2.z = box.z - (box.size_z)/2;

          points_3.x = box.x + (box.size_x)/2;
          points_3.y = box.y + (box.size_y)/2;
          points_3.z = box.z + (box.size_z)/2;

          points_4.x = box.x - (box.size_x)/2;
          points_4.y = box.y - (box.size_y)/2;
          points_4.z = box.z + (box.size_z)/2;
        }