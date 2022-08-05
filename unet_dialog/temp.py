        img = x.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        n, c, w, h = img.shape
        n = int(n ** 0.5)

        img = img[:n * n]
        img = img.reshape(n, n, c, h, w).transpose(0, 3, 1, 4, 2).reshape(n * h, n * w, c)
        img = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR)

        y_pred = y_pred[:n * n]
        y_pred = y_pred.reshape(n, n, 1, h, w).transpose(0, 3, 1, 4, 2).reshape(n * h, n * w, 1)
        y_pred = cv2.cvtColor(np.uint8(y_pred * 255), cv2.COLOR_GRAY2BGR)

        # img = cv2.addWeighted(img, 0.5, y_pred, 0.5, 0)

        y_pred = cv2.threshold(y_pred[:, :, 0], 128, 255, 0)[1]
        contours = cv2.findContours(y_pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        for i in range(len(contours)):
            cv2.drawContours(img, contours, i, (0, 255, 0), 1, 16)
