
        if len(self.model.loss) >= 2:
            w, h = self.label_graph.width(), self.label_graph.height()
            img = np.full((h, w, 3), 100, np.uint8)
            max = np.max(self.model.loss)
            min = np.min(self.model.loss)
            loss = [(i - min) / (max - min) for i in self.model.loss]
            for i in range(len(loss) - 1):
                cv2.line(img,
                         (int(i / self.model.args.num_epochs * w),
                          int(loss[i] * h) + 10),
                         (int((i + 1) / self.model.args.num_epochs * w),
                          int(loss[i + 1] * h) + 10),
                         (0, 0, 255), 1, 16)
            img = cv2.flip(img, 0)
            cv2.putText(img, 'train loss', (10, h - 10), 0, 0.5, (255, 255, 255), 1, 16)
            self.label_graph.setPixmap(QPixmap.fromImage(QImage(img.data, w, h, w * 3, QImage.Format_BGR888)))
